import os
import shutil
import sys
sys.path.append("..")

from einops import rearrange
from huggingface_hub import hf_hub_download

def get_patch_sae_codes(sae, subset_layer_out, num_patches):
    """
    Process layer activations through the SAE to get sparse codes.
    
    This function encodes the layer activations using the SAE,
    then rearranges the resulting codes into a format suitable for further processing (batch_size, width, height, latent_dim).
    
    Parameters
    ----------
    subset_layer_out : torch.Tensor
        The layer activations to encode, typically of shape (batch_size * width * height, feature_dim).
    num_patches : int
        The number of patches in each dimension (width/height) of the image. We always work with square images.
        
    Returns
    -------
    torch.Tensor
        The encoded sparse codes, reshaped to (batch_size, width, height, latent_dim).
    """

    subset_codes = sae.encode(subset_layer_out)
    subset_codes = rearrange(subset_codes.detach(), '(n w h) d -> n w h d',  w=num_patches, h=num_patches)
    subset_codes = subset_codes.float()
    return subset_codes


def resolve_sae_path_from_hub_name(
    sae_name: str,
    data_dir: str,
    hf_org: str = "javifer",
) -> str:
    """
    Resolve a Hub-style SAE id to a local directory containing ae.pt (under .../trainer_0).

    ``sae_name`` is e.g.
    ``google_gemma-3-4b-it-saes/enc_res_out_layer_16_top_k_8192_25_1.0_HF_22191080``,
    matching paths uploaded to ``{hf_org}/google_gemma-3-4b-it-saes`` on the Hub.

    If ``data_dir/saes/<sae_name>/trainer_0`` already has the weights, returns that path;
    otherwise downloads required files from the Hub repo ``{hf_org}/<first_segment>``.
    """
    sae_name = sae_name.strip().replace("\\", "/").strip("/")
    parts = sae_name.split("/")
    if len(parts) < 2:
        raise ValueError(
            "sae_name must look like '<collection>/run', e.g. "
            "google_gemma-3-4b-it-saes/enc_res_out_layer_16_top_k_8192_25_1.0_HF_22191080"
        )
    collection = parts[0]
    path_in_repo = "/".join(parts[1:])
    repo_id = f"{hf_org}/{collection}"

    trainer_dir = os.path.join(data_dir, "saes", *parts, "trainer_0")
    ae_path = os.path.join(trainer_dir, "ae.pt")
    config_path = os.path.join(trainer_dir, "config.json")

    if os.path.isfile(ae_path) and os.path.isfile(config_path):
        return trainer_dir

    os.makedirs(trainer_dir, exist_ok=True)

    def pull(filename: str, required: bool) -> None:
        repo_file = f"{path_in_repo}/{filename}"
        dest = os.path.join(trainer_dir, filename)
        if os.path.isfile(dest):
            return
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                repo_type="model",
            )
            shutil.copy2(cached, dest)
        except Exception as e:
            if required:
                raise RuntimeError(
                    f"Failed to download {repo_file} from {repo_id}: {e}"
                ) from e

    pull("ae.pt", required=True)
    pull("config.json", required=True)
    pull("latent_activation_frequency.json", required=False)
    pull("eval_results.json", required=False)

    if not os.path.isfile(ae_path) or not os.path.isfile(config_path):
        raise RuntimeError(f"SAE download incomplete under {trainer_dir}")
    return trainer_dir