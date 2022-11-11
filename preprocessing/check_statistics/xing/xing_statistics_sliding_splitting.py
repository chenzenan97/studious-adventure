import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

PATH = '../../../data/xing/xing2016/'
FILE = 'interactions'

# keys
USER_KEY='user_id'
ITEM_KEY='item_id'
TIME_KEY='created_at'
SESSION_KEY='session_id'
TYPE_KEY='interaction_type'
# filters
SESSION_THRESHOLD = 30 * 60
MIN_ITEM_SUPPORT = 5  # 20
MIN_SESSION_LENGTH = 2  # 3
MIN_USER_SESSIONS = 3  # 5
MAX_USER_SESSIONS = None  # 200
REPEAT = False  # apply filters several times
CLEAN_TEST = True
SLICES_NUM = 5
SLICE_INTERVAL = 16  # total_interval = 82
DAYS_OFFSET = 0


def make_sessions(data, session_th=SESSION_THRESHOLD, is_ordered=False, user_key=USER_KEY, time_key=TIME_KEY, session_key=SESSION_KEY):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data[user_key].values[1:] != data[user_key].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data[session_key] = session_ids
    return data


def slice_data(data, num_slices=SLICES_NUM, days_offset=DAYS_OFFSET, days_shift=SLICE_INTERVAL):
    for slice_id in range(0, num_slices):
        split_data_slice(data, slice_id, days_offset, days_shift)


def split_data_slice(data, slice_id, days_offset, days_shift):
    start_day = days_offset + (slice_id * days_shift)
    start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(start_day)
    end_day = days_offset + ((slice_id+1) * days_shift)
    end = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc) + timedelta(end_day)

    # prefilter the timespan
    session_max_times = (data.groupby([SESSION_KEY])[TIME_KEY]).max()
    greater_start = session_max_times[session_max_times >= start.timestamp()].index
    lower_end = session_max_times[session_max_times <= end.timestamp()].index
    data = data[np.in1d(data[SESSION_KEY], greater_start.intersection(lower_end))]

    print('Slice data set {}\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(slice_id, len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 start.date().isoformat(), end.date().isoformat()))

    print('--------------------- Slice-Original---')
    report_statistics(data)

    data = filter_data(data)

    # training-test split
    split_data(data, MIN_SESSION_LENGTH)


def last_session_out_split(data, min_session_length):
    """
    last-session-out split
    assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[USER_KEY, TIME_KEY]).groupby(USER_KEY)[SESSION_KEY]
    last_session = sessions.last()
    train = data[~data[SESSION_KEY].isin(last_session.values)].copy()
    test = data[data[SESSION_KEY].isin(last_session.values)].copy()
    if CLEAN_TEST:
        train_items = train[ITEM_KEY].unique()
        test = test[test[ITEM_KEY].isin(train_items)]
        #  remove sessions in test shorter than min_session_length
        slen = test[SESSION_KEY].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test[test[SESSION_KEY].isin(good_sessions)].copy()
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
    return train, test


def split_data(data, min_session_length): #TODO: extend for supproting more than one sessions per use for test
    """
        assign the last session of every user to the test set and the remaining ones to the training set
        """
    train_full_sessions, test_sessions = last_session_out_split(data, min_session_length)
    train_valid_sessions, valid_sessions = last_session_out_split(train_full_sessions, min_session_length)

    print('--------------------- Training---')
    data = train_full_sessions
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)
    print('Training data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    report_statistics(data)

    print('--------------------- Test---')
    data = test_sessions
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)
    print('Test data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    report_statistics(data)

    print('--------------------- Validation_training---:')
    data = train_valid_sessions
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)
    print('Validation_training data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    report_statistics(data)

    print('--------------------- Validation_test---')
    data = valid_sessions
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)
    print('Validation_test data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(), data_end.date().isoformat()))
    report_statistics(data)


def filter_data(data):
    # drop duplicate interactions within the same session
    # data.drop_duplicates(subset=[ITEM_KEY, SESSION_KEY, TYPE_KEY], keep='first', inplace=True)
    condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
        [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
        [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
    count = 1
    while not condition:
        print(count)
        # keep items with >=20 interactions
        item_pop = data[ITEM_KEY].value_counts()
        good_items = item_pop[item_pop >= MIN_ITEM_SUPPORT].index
        data = data[data[ITEM_KEY].isin(good_items)]
        # remove sessions with length < 3
        session_length = data[SESSION_KEY].value_counts()
        good_sessions = session_length[session_length >= MIN_SESSION_LENGTH].index
        data = data[data[SESSION_KEY].isin(good_sessions)]
        # let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
        sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
        if MAX_USER_SESSIONS is None:  # no filter for max number of sessions for each user
            good_users = sess_per_user[(sess_per_user >= MIN_USER_SESSIONS)].index
        else:
            good_users = sess_per_user[(sess_per_user >= MIN_USER_SESSIONS) & (sess_per_user < MAX_USER_SESSIONS)].index
        data = data[data[USER_KEY].isin(good_users)]
        condition = data.groupby(USER_KEY)[SESSION_KEY].nunique().min() >= MIN_USER_SESSIONS and data.groupby(
            [USER_KEY, SESSION_KEY]).size().min() >= MIN_SESSION_LENGTH and data.groupby(
            [ITEM_KEY]).size().min() >= MIN_ITEM_SUPPORT
        count += 1
        if not REPEAT:
            break

    # output
    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Filtered data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(),
                 data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Slice-Filtered---')
    report_statistics(data)

    return data


def report_statistics(data):
    print('--------------------- Statistics---')
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    print('---------------------')
    # print('Num of users: {}'.format(data[USER_KEY].nunique()))
    # print('Max num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().max()))
    # print('Min num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().min()))
    # print('Median num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().median()))
    # print('Mean num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().mean()))
    # print('Std num of users\' interactions: {}'.format(data.groupby([USER_KEY]).size().std()))
    sess_per_user = data.groupby(USER_KEY)[SESSION_KEY].nunique()
    print('Max num of users\' sessions: {}'.format(sess_per_user.max()))
    print('Min num of users\' sessions: {}'.format(sess_per_user.min()))
    print('Median num of users\' sessions: {}'.format(sess_per_user.median()))
    print('Mean num of users\' sessions: {}'.format(sess_per_user.mean()))
    print('Std num of users\' sessions: {}'.format(sess_per_user.std()))
    print('---------------------')
    # print('Num of sessions per user: {}'.format(np.count_nonzero(data.groupby(USER_KEY)[SESSION_KEY].nunique())))
    print('Max sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().max()))
    print('Min sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().min()))
    print('Median sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().median()))
    print('Mean sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().mean()))
    print('Std sessions\' length: {}'.format(data.groupby([USER_KEY, SESSION_KEY]).size().std()))
    print('---------------------')
    # print('Num of items: {}'.format(data[ITEM_KEY].nunique()))
    # print('Max num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().max()))
    # print('Min num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().min()))
    # print('Median num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().median()))
    # print('Mean num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().mean()))
    # print('Std num of interactions done with an item: {}'.format(data.groupby([ITEM_KEY]).size().std()))

    print('Max num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().max()))
    print('Min num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().min()))
    print('Median num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().median()))
    print('Mean num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().mean()))
    print('Std num of interactions done with an item: {}'.format(data[ITEM_KEY].value_counts().std()))
    print('---------------------')


if __name__ == '__main__':

    # updater.dispatcher.add_handler( CommandHandler('status', status) )
    data = pd.read_csv(PATH + FILE + '.csv', sep='\t')
    # remove interactions of type 'delete'
    data = data[data[TYPE_KEY] != 4].copy()

    # partition interactions into sessions with 30-minutes idle time
    data = make_sessions(data)

    data_start = datetime.fromtimestamp(data[TIME_KEY].min(), timezone.utc)
    data_end = datetime.fromtimestamp(data[TIME_KEY].max(), timezone.utc)

    print('Original data set\n\tEvents: {}\n\tUsers: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format(len(data), data[USER_KEY].nunique(), data[SESSION_KEY].nunique(), data[ITEM_KEY].nunique(), data_start.date().isoformat(),
                 data_end.date().isoformat()))

    print('--------------------- Original---')
    report_statistics(data)

    slice_data(data)

