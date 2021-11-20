import pytest
import pandas as pd
import numpy as np
import mock
from ml_versioning_wrapper.get_data import get_data_from_storage
from ml_versioning_wrapper.commit import get_url_push


@pytest.fixture
def git_remote_show_message():
    return "remote origin \n" \
           "Fetch URL: https://github.com/example \n" \
           "Push  URL: https://github.com/example \n" \
           "HEAD branch: main \n" \
           "Remote branch: \n" \
           "  main tracked \n" \
           "Local branch configured for 'git pull': \n" \
           "  main merges with remote main \n" \
           "Local ref configured for 'git push':\n" \
           "  main pushes to main (up to date)\n"


@mock.patch('ml_versioning_wrapper.commit.git.cmd.Git')
def test_get_url_push(mock, git_remote_show_message):
    mock.return_value.execute.return_value = git_remote_show_message
    remote_url = get_url_push("current_work_directory")
    assert remote_url == "https://github.com/example"
