import numpy as np
import datetime

from utilities import str2date



def get_issue_transitions( the_issue ):
    
    """get all transitions that an issue has made


    Parameters
    ----------
    the_issue : jira issue
        issue returned from e.g. a jira server

    Returns
    -------
    list of jira transitions
        each has the fields 'date', 'from', 'to'
    """
    
    transitions = [];
    
    for history in the_issue.changelog.histories:
        for item in history.items:
            if item.field == 'status':
                transitions.append({})
                date_created = str2date( history.created )
                transitions[ len(transitions)-1 ]['date'] = date_created
                transitions[ len(transitions)-1 ]['from'] = item.fromString
                transitions[ len(transitions)-1 ]['to']   = item.toString
                
    return transitions


def time_in_status_in_hours( transitions, 
                            str_status ):
    """compute the time spent in a certain status from a set of transitions
    
    transitions only contain the moments in time when the issue was
    transitioned, *not the duration spent in a certain status

    Parameters
    ----------
    transitions : set of jira issue transitions
        computed from a jira issue e.g. via get_issue_transitions
        
    Returns
    -------
    numpy scalar
        time spent in the respective status in hours
    """
    np_time    = np.empty(0)
    date_start = None
    date_end   = None
    for idx, transition in enumerate( transitions ):
        if transition['to'] == str_status:
            date_start = transition['date']
            if idx+1 < len(transitions):
                date_end = transitions[idx+1]['date']
            else:
                date_end = datetime.datetime.now()
            diff = date_end - date_start
            np_time = np.append( np_time, diff.days * 24 + diff.seconds/3600 )
    if np_time.size == 0:
        np_time = np.zeros(1)
    return np_time.sum()


            



def download_data( jira_connection, 
                  str_output_file, 
                  list_of_jira_project_prefixes, 
                  str_status_to_look_for):
    """downloads issues from a jira server and saves them to a CSV file
    
    all issues are saved that match a list of project prefixes

    Parameters
    ----------
    jira_connection : jira connection
        connection to a jira server
        
    str_output_file : string
        (path and) filename of the output file
        
    list_of_jira_project_prefixes : list of strings
        e.g. ['PROJ1', 'PROJ2']
    
    str_status_to_look_for : string
        name of the status to look for
        e.g. "In Progress"
        
    Returns
    -------
    void
    """
    
    with open(str_output_file, "w", encoding="utf-8") as f:
        
        #write header
        f.write("index"          + ", " + 
                "project_prefix" + ", " + 
                "issuekey"       + ", " + 
                "issuetype"      + ", " + 
                "priority"       + ", " + 
                "reporter"       + ", " + 
                "timeInProgress" + ", " + 
                "resolution"     + ", " + 
                "summary"        + ", " + 
                "description"    + "\n" )
        
        counter = -1
        
        for prefix in list_of_jira_project_prefixes:
    
            step = 250
            
            for idx in range(1, 25000, step):
                
                str_jql_query = "issuekey in ("
                for k in range(idx, idx+step):
                    str_jql_query += prefix + "-" + str(k)
                    if k != idx+step-1:
                        str_jql_query += ", "
                        
                str_jql_query += ")"
                                
                tmp = jira_connection.search_issues(str_jql_query, startAt=0, maxResults=step, validate_query=True, fields=None, expand='changelog', json_result=None)
                
                for issue in tmp[::-1]:
                    
                    if False:
                
                        str_issue_key = prefix + "-"+str(idx)
                        
                        try:
                            issue = jira_connection.issue(str_issue_key, expand='changelog')
                        except:
                            continue
                
                        
                    transitions = get_issue_transitions(issue)
                    hours_in_progress = time_in_status_in_hours( transitions, str_status = str_status_to_look_for)        
                    
                    print("Issue " + issue.key.ljust(16) + " was " + "%7.1fh"%(hours_in_progress) + " in Status 'In Progress'")

                    
                    # write to file
                    counter += 1
                    
                    new_entry = ""
                    new_entry += "%7d"%(counter) + ","
                    new_entry += (prefix + ",").ljust(16) 
                    new_entry += (issue.key + ",").ljust(16) 
                    new_entry += (issue.fields.issuetype.name + ",").ljust(16)
                    
                    try:
                        new_entry += (issue.fields.priority.name + ",").ljust(16) 
                    except:
                        new_entry += "None,"
                            
                    new_entry += (str(issue.fields.reporter) + ",").ljust(16) 
                    new_entry += "%7.1f"%(hours_in_progress) + "," 
                    
                    try:
                        new_entry += str(issue.fields.resolution) + ","
                    except:
                        new_entry += "None,"
                        
                    new_entry += issue.fields.summary.replace(",", "").replace("\n", " ").replace("\r", " ") + "," 
                    new_entry += str(issue.fields.description).replace(",", "").replace("\n", " ").replace("\r", " ")
                    new_entry +=  "\n"
                    f.write( new_entry )

                    
                
    