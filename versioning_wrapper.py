import git
import os
import mlflow
import pickle


def versioning_wrapper(func):
    def wrapper(branch:str, gitwd:str, *args, **kwargs):
        ##########################
        # Set repo git in python #
        ##########################
        repo = git.Repo(gitwd)
        local_branch = repo.active_branch.name
        try:
            assert local_branch == branch
        except AssertionError:
            raise AssertionError(f"You are not in branch {branch}. "
                                 f"Just checkout to this branch and try again!")

        #####################
        # Set mlflow params #
        #####################
        experiment_name = 'default_model'
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():

            ####################
            # train your model #
            ####################

            results = func(*args, **kwargs)

            ####################
            # log your results #
            ####################

            for object in results.keys():
                obj = results[object]
                if isinstance(obj, dict):
                    mlflow.log_metrics(obj)
                else:
                    pickle.dump(obj, open(f"{object}.pkl", 'wb'))
                    mlflow.log_artifact(f"{object}.pkl")
                    os.remove(f"{object}.pkl")

            run = mlflow.active_run()
            run_id = run.info.run_id

        ###############
        # Commit code #
        ###############

        files = repo.git.diff(None, name_only=True)
        for f in files.split('\n'):
            repo.git.add(f)

        repo.git.commit("-m", f"exp(run_id) {run_id}")

        return results

    return wrapper

