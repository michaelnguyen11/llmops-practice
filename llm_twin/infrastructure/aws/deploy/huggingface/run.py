from loguru import logger

try:
    from sagemaker.enums import EndpointType
    from sagemaker.huggingface import get_huggingface_llm_image_uri
except ModuleNotFoundError:
    logger.warning("Couldn't load SageMaker imports.")

from llm_twin.model.utils import ResourceManager
from llm_twin.settings import settings
from .config import hugging_face_deploy_config, model_resource_config
from .sagemaker_huggingface import DeploymentService, SagemakerHuggingfaceStrategy


def create_endpoint(endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED):
    assert settings.AWS_ARN_ROLE, "AWS ARN role is required."
    logger.info(
        f"Creating endpoint with endpoint_type = {endpoint_type} and model_id = {settings.HF_MODEL_ID}"
    )

    llm_image = get_huggingface_llm_image_uri("huggingface")

    resource_manager = ResourceManager()
    deployment_service = DeploymentService(resource_manager=resource_manager)

    huggingface_deployment = SagemakerHuggingfaceStrategy(deployment_service)
    huggingface_deployment.deploy(
        role_arn=settings.AWS_ARN_ROLE,
        llm_image=llm_image,
        config=hugging_face_deploy_config,
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        endpoint_config_name=settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE,
        gpu_instance_type=settings.GPU_INSTANCE_TYPE,
        resources=model_resource_config,
        endpoint_type=endpoint_type,
    )


if __name__ == "__main__":
    create_endpoint(endpoint_type=EndpointType.MODEL_BASED)
