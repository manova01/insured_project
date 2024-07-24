from prefect.deployments import Deployment
from prefect.filesystems import GitHub
from your_module import insurance_model_training  # Import the Prefect flow

# Register and run the flow
if __name__ == "__main__":
    insurance_model_training()

    # Deployment to GitHub (optional, if needed for version control and remote execution)
    github_block = GitHub.load("insured-project-github-repo")
    deployment = Deployment.build_from_flow(
        flow=insurance_model_training,
        name="insurance_model_training_deployment",
        storage=github_block,
    )
    deployment.apply()
