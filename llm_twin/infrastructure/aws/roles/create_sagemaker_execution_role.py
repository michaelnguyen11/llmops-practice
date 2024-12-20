import json
from pathlib import Path

from loguru import logger

try:
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports.")

from llm_twin.settings import settings


def create_sagemaker_execution_role(role_name: str):
    assert settings.AWS_REGION, "AWS_REGION is not set."
    assert settings.AWS_ACCESS_KEY, "AWS_ACCESS_KEY is not set."
    assert settings.AWS_SECRET_KEY, "AWS_SECRET_KEY is not set."

    # Create an IAM client
    iam = boto3.client(
        "iam",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
    )

    # Define the trust relationship policy
    trust_relationship = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        # Create the IAM role
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description="IAM role for SageMaker execution",
        )

        # Attach neccessary policies
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        ]

        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        logger.info(f"Role '{role_name}' created successfully.")
        logger.info(f"Role ARN: {role['Role']['Arn']}")

        return role["Role"]["Arn"]

    except iam.exceptions.EntityAlreadyExistsException:
        logger.warning(f"Role '{role_name}' already exists. Fetching its ARN...")
        role = iam.get_role(RoleName=role_name)
        logger.info(f"Using existing role ARN: {role['Role']['Arn']}")

        return role["Role"]["Arn"]

    except Exception as e:
        logger.error(f"Failed to create IAM role '{role_name}': {e}")
        raise


if __name__ == "__main__":
    role_name = "SageMakerExecutionRoleLLM"
    role_arn = create_sagemaker_execution_role(role_name)
    print(f"SageMaker execution role ARN: {role_arn}")

    # Save the role ARN to a file
    with Path("sagemaker_execution_role.json").open("w") as f:
        json.dump({"RoleArn": role_arn}, f)

    logger.info("Role ARN saved to 'sagemaker_execution_role.json'")
