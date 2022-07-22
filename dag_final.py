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

# These imports are not ready yet.
# from prefect.tasks.toloka import Answer, get_answers


DB = '/tmp/prefect_example.db'


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
