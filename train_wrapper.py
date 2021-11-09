from wrapper_tools.commit import check_branch, commit_code
import os
import mlflow
import pickle


def train_wrapper(func):
    def wrapper(branch: str, gitwd: str, *args, **kwargs):
        ##########################
        # Set repo git in python #
        ##########################
        repo = check_branch(branch, gitwd)

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

        commit_code(repo, f"exp(train): run_id={run_id}")

        return results

    return wrapper
