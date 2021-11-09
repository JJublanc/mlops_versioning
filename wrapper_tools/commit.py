import git


def check_branch(branch: str, gitwd: str):
    """
    Check if the current branch is the same as the one specified. Return the repo if it is.
    :param branch: branch name
    :param gitwd: working directory where .git folder is located
    :return:
    """
    repo = git.Repo(gitwd)
    local_branch = repo.active_branch.name
    try:
        assert local_branch == branch
    except AssertionError:
        raise AssertionError(f"You are not in branch {branch}. "
                             f"Just checkout to this branch and try again!")
    return repo


def commit_code(repo: git.Repo, commit_msg: str):
    """
    Commit code to the repo
    :param repo: git repo
    :param commit_msg: message to commit
    :return: None
    """
    files = repo.git.diff(None, name_only=True)
    for f in files.split('\n'):
        repo.git.add(f)

    repo.git.commit("-m", commit_msg)
