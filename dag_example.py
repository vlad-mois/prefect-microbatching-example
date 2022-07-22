import attr
import logging
import pandas as pd
import prefect
import sqlite3
import time
from contextlib import closing
from crowdkit.aggregation import MajorityVote
from datetime import timedelta
from prefect import Flow, Parameter, task, unmapped
from prefect.tasks.control_flow.case import case
from prefect.tasks.core.operators import NotEqual
from prefect.tasks.toloka import accept_assignment, reject_assignment
from toloka.client import Pool, Task as TolokaTask, TolokaClient
from toloka.streaming.cursor import AssignmentCursor
from toloka_prefect.utils import with_logger, with_toloka_client
from typing import Dict, List, Optional


DB = '/tmp/prefect_example.db'

# Run this before launching the DAG.
conn = sqlite3.connect(DB)
cursor = conn.cursor()
cursor.executescript('''
    DROP TABLE IF EXISTS images;
    CREATE TABLE IF NOT EXISTS images (
        url             TEXT PRIMARY KEY,
        created_ts      DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        started_ts      DATETIME,
        labeled_ts      DATETIME,
        processed_cnt   INTEGER DEFAULT 0 NOT NULL,
        flow_run_id     CHAR(36),
        label           TEXT
    );

    INSERT INTO images (url) VALUES
        ("https://avatars.mds.yandex.net/get-canvas/5415150/2a00000181cfe136a5cbc20fb993e408bf60/cropSource"),
        ("https://avatars.mds.yandex.net/get-canvas/5405791/2a00000181cfc24a87a64dc392e7b3490c15/cropSource"),
        ("https://avatars.mds.yandex.net/get-canvas/5401683/2a00000181cfb816b1f871790ef154a1d388/cropSource"),
        ("https://avatars.mds.yandex.net/get-canvas/5405791/2a00000181cfb206e0148a0ffe926f8d059a/cropSource"),
        ("https://avatars.mds.yandex.net/get-canvas/5415150/2a00000181cfd581ac9e60e0a4712adb3b41/cropSource")
    ;
''')
conn.commit()
cursor.close()
conn.close()
del conn
#


TEMPLATE_QUERY_MOVE_TO_PROCESSING = '''
    UPDATE images
    SET
        flow_run_id = "{flow_run_id}",
        processed_cnt = processed_cnt + 1,
        started_ts = CURRENT_TIMESTAMP
    WHERE url IN (
        SELECT url FROM images WHERE flow_run_id IS NULL {limit}
    )
'''
TEMPLATE_QUERY_GET_IMAGES = 'SELECT url FROM images WHERE flow_run_id = "{flow_run_id}"'
TEMPLATE_QUERY_SET_LABELS = '''
    INSERT INTO images (url, label)
    VALUES (?, ?)
    ON CONFLICT (url) DO
    UPDATE SET
        label = excluded.label,
        labeled_ts = CURRENT_TIMESTAMP
'''
TEMPLATE_QUERY_RELAUNCH = '''
    INSERT INTO images (url)
    VALUES (?)
    ON CONFLICT (url) DO
    UPDATE SET
        flow_run_id = NULL
'''

MESSAGE_ACCEPT = 'Well done'
MESSAGE_REJECT = 'Incorrect object'


@task
@with_logger
def move_into_processing(limit: Optional[int] = None, *, logger: logging.Logger) -> None:
    flow_run_id = prefect.context['flow_run_id']
    query = TEMPLATE_QUERY_MOVE_TO_PROCESSING.format(flow_run_id=flow_run_id, limit=f'LIMIT {limit}' if limit else '')
    logger.info('Running query: %s', query)
    with closing(sqlite3.connect(DB)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.executescript(query)
            conn.commit()


@task
@with_logger
def get_images(*, logger: logging.Logger) -> List[str]:
    flow_run_id = prefect.context['flow_run_id']
    query = TEMPLATE_QUERY_GET_IMAGES.format(flow_run_id=flow_run_id)
    logger.info('Running query: %s', query)
    with closing(sqlite3.connect(DB)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(query)
            res = cursor.fetchall()
            conn.commit
            return [item[0] for item in res]


@task
@with_toloka_client
@with_logger
def create_tasks(
    images: List[str],
    pool_id: str,
    *,
    logger: logging.Logger,
    toloka_client: TolokaClient,
) -> List[TolokaTask]:
    tasks_prepared = [TolokaTask(input_values={'image': url}, pool_id=pool_id) for url in images]
    res = toloka_client.create_tasks(tasks_prepared, allow_defaults=True, open_pool=True)
    logger.info('Created tasks count: %d', len(res.items))
    return list(res.items.values())


@attr.s(frozen=True)
class Answer:
    task_id: str = attr.ib()
    assignment_id: str = attr.ib()
    input_values: Dict = attr.ib()
    user_id: Optional[str] = attr.ib(default=None)
    output_values: Optional[Dict] = attr.ib(default=None)


@task
@with_toloka_client
@with_logger
def get_answers(
    tasks: List[TolokaTask],
    *,
    period: timedelta = timedelta(minutes=1),
    overlap: Optional[int] = None,
    logger: logging.Logger,
    toloka_client: TolokaClient,
) -> List[Answer]:
    pool_id = tasks[0].pool_id
    start_time = tasks[0].created
    cursor_kwargs = {'pool_id': pool_id, 'created_gte': start_time, 'toloka_client': toloka_client}
    it_submitted = AssignmentCursor(event_type='SUBMITTED', **cursor_kwargs)
    logger.info('Start from: %s', start_time)

    remaining_by_task: Dict[str, int] = {task.id: overlap or task.overlap for task in tasks}
    result = []
    while True:
        new_answers = []
        for event in it_submitted:
            for _task, solution in zip(event.assignment.tasks, event.assignment.solutions):
                if _task.id in remaining_by_task:
                    remaining_by_task[_task.id] = max(remaining_by_task[_task.id] - 1, 0)
                    new_answers.append(Answer(_task.id,
                                              event.assignment.id,
                                              _task.input_values,
                                              event.assignment.user_id,
                                              solution.output_values))
        result.extend(new_answers)
        logger.info('New answers submitted count: %d. Total count: %d', len(new_answers), len(result))

        finished_completely_count = sum(1 for remaining in remaining_by_task.values() if not remaining)
        logger.info('Finished tasks count: %d', finished_completely_count)
        logger.debug('Remaining answers count by task: %s', remaining_by_task)

        if finished_completely_count == len(remaining_by_task):
            return result
        elif not new_answers:
            pool = toloka_client.get_pool(pool_id)
            if pool.status != Pool.Status.OPEN:
                raise ValueError(f'Waiting for pool {pool_id} in status {pool.status.value}')
        logger.info('Sleep for %d seconds', period.total_seconds())
        time.sleep(period.total_seconds())


@task
@with_toloka_client
@with_logger
def create_val_tasks(
    answers: List[Answer],
    val_pool_id: str,
    *,
    logger: logging.Logger,
    toloka_client: TolokaClient,
) -> List[TolokaTask]:
    tasks_prepared = [TolokaTask(pool_id=val_pool_id,
                                 unavailable_for=[answer.user_id],
                                 input_values={'image': answer.input_values['image'],
                                               'found_link': answer.output_values['found_link'],
                                               'assignment_id': answer.assignment_id})
                      for answer in answers
                      if answer.output_values is not None]
    res = toloka_client.create_tasks(tasks_prepared, allow_defaults=True, open_pool=True)
    logger.info('Created validation tasks count: %d', len(res.items))
    return list(res.items.values())


@task
def aggregate(val_answers: List[Answer]) -> Dict[str, str]:
    tuples = [(answer.input_values['assignment_id'], answer.output_values['result'], answer.user_id)
              for answer in val_answers
              if answer.output_values is not None]
    df = pd.DataFrame(tuples, columns=('task', 'label', 'worker'))
    return MajorityVote().fit_predict(df)


@task
def select_accepted(aggregated: Dict[str, str]) -> List[str]:
    return [assignment_id for assignment_id, result in aggregated.items() if result == 'Yes']


@task
def select_rejected(aggregated: Dict[str, str]) -> List[str]:
    return [assignment_id for assignment_id, result in aggregated.items() if result != 'Yes']


@task
def save_results(accepted: List[str], answers: List[Answer]) -> None:
    accepted = set(accepted)
    with closing(sqlite3.connect(DB)) as conn:
        with closing(conn.cursor()) as cursor:
            data = [(answer.input_values['image'], answer.output_values['found_link'])
                    for answer in answers
                    if answer.assignment_id in accepted]
            cursor.executemany(TEMPLATE_QUERY_SET_LABELS, data)
            conn.commit()


@task
def relaunch_non_accepted(accepted: List[str], answers: List[Answer]) -> None:
    accepted = set(accepted)
    accepted_images = {answer.input_values['image'] for answer in answers if answer.assignment_id in accepted}
    with closing(sqlite3.connect(DB)) as conn:
        with closing(conn.cursor()) as cursor:
            data = [(answer.input_values['image'],)
                    for answer in answers
                    if answer.input_values['image'] not in accepted_images]
            cursor.executemany(TEMPLATE_QUERY_RELAUNCH, data)
            conn.commit()


with Flow('microbatching-flow') as flow:
    pool_id = Parameter('pool_id', '34278011')
    val_pool_id = Parameter('val_pool_id', '34278029')

    _moved = move_into_processing(limit=4)
    images = get_images(upstream_tasks=[_moved])
    with case(NotEqual()(images, []), True):
        tasks = create_tasks(images, pool_id)
        answers = get_answers(tasks)
        val_tasks = create_val_tasks(answers, val_pool_id)
        val_answers = get_answers(val_tasks)
        aggregated = aggregate(val_answers)
        to_accept = select_accepted(aggregated)
        to_reject = select_rejected(aggregated)
        accepted = accept_assignment.map(to_accept, unmapped(MESSAGE_ACCEPT))
        reject_assignment.map(to_reject, unmapped(MESSAGE_REJECT))
        save_results(accepted, answers)
        relaunch_non_accepted(accepted, answers)

flow.register(project_name='Some test project')
