import os
import numpy as np
import pandas as pd 
import datetime
from time import time
from tqdm import tqdm
from collections import Counter
import pickle
import json
from pandas.io.json import json_normalize

try:
	os.chdir(os.path.join(os.getcwd(), 'data_science_bowl_2019/notebooks/data'))
	print(os.getcwd())
except:
	pass


# Load datasets
train = pd.read_csv('data/raw/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('data/raw/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('data/raw/data-science-bowl-2019/test.csv')
#specs = pd.read_csv('input/raw/data-science-bowl-2019/specs.csv')


# Filter unusefull data
keep_id = train[train.type == "Assessment"][["installation_id"]].drop_duplicates()
train = pd.merge(train, keep_id, on="installation_id", how="inner")

print("Train: ",train.shape)
print("Number of id we keep: ",keep_id.shape)

# installation_id's who did assessments (we have already taken out the ones who never took one),
#  but without results in the train_labels? As you can see below, yes there are 628 of those.
discard_id = train[train.installation_id.isin(train_labels.installation_id.unique()) != True].installation_id.unique()

print("Number of ids that were discarted: ",discard_id.shape[0])

train = train[train.installation_id.isin(discard_id)!=True]


# Basically what we need to do is to compose aggregated features for each session of which we know the train label.
print(f'Number of rows in train_labels: {train_labels.shape[0]}')
print(f'Number of unique game_sessions in train_labels: {train_labels.game_session.nunique()}')

#  [markdown]
# ### Fix num correct and incorrect variables 
#  [markdown]
# geting from data-science-bowl-2019-data-exploration
#  [markdown]
# From Kaggle: The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains "correct":true.
# 
# However, in the first version I already noticed that I had one attempt too many for this installation_id when mapping the rows with the train_labels for. It turns out that there are in fact also assessment attemps for Bird Measurer with event_code 4100, which should not count (see below). In this case that also makes sense as this installation_id already had a pass on the first attempt

# 
#credits for this code chuck go to Andrew Lukyanenko
train['attempt'] = 0
train.loc[(train['title'] == 'Bird Measurer (Assessment)') & (train['event_code'] == 4110),       'attempt'] = 1
train.loc[(train['type'] == 'Assessment') &       (train['title'] != 'Bird Measurer (Assessment)')       & (train['event_code'] == 4100),          'attempt'] = 1

train['correct'] = None
train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":true')), 'correct'] = True
train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":false')), 'correct'] = False

# =============================================================== FUNCTIONS ============================================================================================= #


def add_datepart(df: pd.DataFrame, field_name: str,
                 prefix: str = None, drop: bool = True, time: bool = True, date: bool = True):
    """
    Helper function that adds columns relevant to a date in the column `field_name` of `df`.
    from fastai: https://github.com/fastai/fastai/blob/master/fastai/tabular/transform.py#L55
    """
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end', 'Is_month_start']
    if date:
        attr.append('Date')
    if time:
        attr = attr + ['Hour', 'Minute']
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


#Credits go to Andrew Lukyanenko

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code


def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    user_activities_even_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    user_activities_lasttime = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}


    # news features: time spent in each activity
    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}
    accumulated_act_count = {actv: 0 for actv in list_of_user_activities}

    event_code_count = {eve: 0 for eve in list_of_event_code}
    last_session_time_sec = 0
    
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0 
    accumulated_actions = 0
    accumulated_correct_games = 0
    accumulated_uncorrect_games = 0
    accumulated_misses_games = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title] #from Andrew

        # get num event per session
        event_count = session['event_count'].iloc[-1]
        
        # get current session time in seconds
        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_act[activities_labels[session_title]] += time_spent
            accumulated_act_count[activities_labels[session_title]] +=1

            # get last time of each type of session
            user_activities_lasttime[session_type] = session['timestamp'].iloc[-1]


            
        # extract info from event_data. We must parcer json info to get it
        if session_type == 'Game':
            event_data = []
            
            for ev in session['event_data']:
                event_data.append(json.loads(ev))
            
            event_data = pd.DataFrame(event_data)
           
            if 'correct' in event_data.columns: 
                correct_games = Counter(event_data.correct)

                accumulated_correct_games += correct_games[True]
                accumulated_uncorrect_games += correct_games[False]
            
            if 'misses' in event_data.columns: 
                accumulated_misses_games += np.sum(event_data.misses)
    


        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(time_spent_each_act.copy())
            features.update(event_code_count.copy())

            act_count_tmp = {k + '_count': v for k,v in accumulated_act_count.items()}
            features.update(act_count_tmp)
            del act_count_tmp

            user_activities_even_count_tmp = {k + '_eventacum': v for k,v in user_activities_even_count.items()}
            features.update(user_activities_even_count_tmp)
            del user_activities_even_count_tmp

             # diff time activities features
            user_activities_diff_time = {'Assessment_Clip_difftime':0, 'Assessment_Activity_difftime': 0, 
            'Assessment_Assessment_difftime': 0, 'Assessment_Game_difftime':0}

            if user_activities_lasttime['Clip'] != 0:
                user_activities_diff_time['Assessment_Clip_difftime'] = (session['timestamp'].iloc[-1] - user_activities_lasttime['Clip']).seconds
            
            if user_activities_lasttime['Activity'] != 0:
                user_activities_diff_time['Assessment_Activity_difftime'] = (session['timestamp'].iloc[-1] - user_activities_lasttime['Activity']).seconds
            
            if user_activities_lasttime['Game'] != 0:
                user_activities_diff_time['Assessment_Game_difftime'] = (session['timestamp'].iloc[-1] - user_activities_lasttime['Game']).seconds

            if user_activities_lasttime['Assessment'] != 0: 
                user_activities_diff_time['Assessment_Assessment_difftime'] = (session['timestamp'].iloc[-1] - user_activities_lasttime['Assessment']).seconds

            features.update(user_activities_diff_time.copy())

            # get last time of assessment
            user_activities_lasttime[session_type] = session['timestamp'].iloc[-1]
            


            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1] #from Andrew
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0] 
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # how many correct and uncorrect game has done so far.
            features['accumulated_correct_games'] = accumulated_correct_games
            features['accumulated_uncorrect_games'] = accumulated_uncorrect_games
            features['accumulated_misses_games'] = accumulated_misses_games

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        n_of_event_codes = Counter(session['event_code'])
        
        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
        
        # counts how many events the player has done so far for each activity type
        user_activities_even_count[session_type] += event_count
    # if test_set=True, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in train_set, all assessments are kept
    return all_assessments


def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        
        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]].sum(axis = 1)
        
        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')

        # calculate diff for each number of actions
        df['Clip_eventacum_diff'] = df['Clip_eventacum'].diff()
        df['Game_eventacum_diff'] = df['Game_eventacum'].diff()
        df['Activity_eventacum_diff'] = df['Activity_eventacum'].diff()
        df['Assessment_eventacum_diff'] = df['Assessment_eventacum'].diff()

        df['accumulated_correct_attempts_diff'] = df['accumulated_correct_attempts'].diff()
        df['accumulated_uncorrect_attempts_diff'] = df['accumulated_uncorrect_attempts'].diff()

        df['accumulated_correct_games_diff'] = df['accumulated_correct_games'].diff()
        df['accumulated_uncorrect_games_diff'] = df['accumulated_uncorrect_games'].diff()
        df['accumulated_misses_games_diff'] = df['accumulated_misses_games'].diff()

        
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]
   
    return reduce_train, reduce_test, features


# ======================================================================================================================================================================= #



# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)

categoricals = ['session_title']



# user_sample = train[train.installation_id == "0006a69f"]
# sample_id_data = get_data(sample_id) #returns a list
# sample_df = pd.DataFrame(sample_id_data)
# sample_df.iloc[:,-10:]


#The get_data function is applied to each installation_id and added to the compile_data list
compiled_data = []
# tqdm is the library that draws the status bar below
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)), total=train.installation_id.nunique(), desc='Installation_id', position=0):
    # user_sample is a DataFrame that contains only one installation_id
    compiled_data += get_data(user_sample)

#Compiled_data is converted into a DataFrame and deleted to save memmory
reduce_train = pd.DataFrame(compiled_data)
del compiled_data

print("Shape train features",reduce_train.shape)

# applying get_data to submit dataset
new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(), desc='Installation_id', position=0):
    a = get_data(user_sample, test_set=True)
    new_test.append(a)
    
reduce_test = pd.DataFrame(new_test)

# call feature engineering function
reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)

# set index
reduce_train.index = reduce_train.installation_id.tolist()
reduce_test.index = reduce_test.installation_id.tolist()

cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

y = reduce_train['accuracy_group'].copy()

cols_to_keep = [c for c in reduce_train.columns if c not in cols_to_drop]
reduce_train = reduce_train[cols_to_keep]
reduce_test = reduce_test[cols_to_keep]


# saving datasets
reduce_train.to_pickle('data/features/train_features_003.pkl')

y.to_pickle('data/features/train_labels.pkl')
reduce_test.to_pickle('data/features/submit_features_003.pkl')