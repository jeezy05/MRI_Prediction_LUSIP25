import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, morphology, measure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
import warnings
warnings.filterwarnings('ignore')

import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor

def process_image_wrapper(args):
    pipeline_config, input_file, rel_prefix = args
    out_dir = os.path.join(pipeline_config['output_dir'], os.path.dirname(rel_prefix))
    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, os.path.splitext(os.path.basename(rel_prefix))[0])
    pipeline = MRIPreprocessingPipeline(
        input_dir=pipeline_config['input_dir'],
        template_dir=pipeline_config['template_dir'],
        output_dir=pipeline_config['output_dir']
    )
    try:
        pipeline.preprocess_single_image(input_file, output_prefix)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

class MRIPreprocessingPipeline:
    def __init__(self, input_dir="IXI-T1", template_dir="templates", output_dir="preprocessed"):
        self.input_dir = input_dir
        self.template_dir = template_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.template_path = os.path.join(template_dir, "mni_icbm152_t1_tal_nlin_sym_09a.nii")
        self.template_mask_path = os.path.join(template_dir, "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii")
        if os.path.exists(self.template_path):
            self.template_img = nib.load(self.template_path)
            self.template_data = self.template_img.get_fdata()
        else:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        if os.path.exists(self.template_mask_path):
            self.template_mask_img = nib.load(self.template_mask_path)
            self.template_mask_data = self.template_mask_img.get_fdata()
        else:
            print("Warning: Template mask not found, will use simple thresholding for skull stripping")
            self.template_mask_data = None

    def load_image(self, file_path):
        return nib.load(file_path)
    
    def denoise_image(self, img_data, method='gaussian'):
        print("Applying denoising...")
        if method == 'gaussian':
            denoised = gaussian_filter(img_data, sigma=1.0)
        elif method == 'tv':
            denoised = denoise_tv_chambolle(img_data, weight=0.1)
        elif method == 'median':
            denoised = ndimage.median_filter(img_data, size=3)
        elif method == 'nonlocal_means':
            from skimage.restoration import denoise_nl_means
            denoised = denoise_nl_means(img_data, h=0.1, fast_mode=True, 
                                      patch_size=5, patch_distance=7)
        elif method == 'bilateral_3d':
            denoised = ndimage.gaussian_filter(img_data, sigma=1.0)
            denoised = ndimage.uniform_filter(denoised, size=3)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return denoised

    def estimate_bias_field(self, img_data, degree=3, method='conservative'):
        print("Estimating bias field...")
        if method == 'conservative':
            return self._estimate_conservative_bias_field(img_data)
        elif method == 'gaussian':
            return self._estimate_gaussian_bias_field(img_data)
        else:
            return self._estimate_polynomial_bias_field(img_data, degree)

    def _estimate_conservative_bias_field(self, img_data):
        bias_field = gaussian_filter(img_data, sigma=20)
        bias_field_mean = np.mean(bias_field)
        bias_field = bias_field / bias_field_mean
        bias_field = np.clip(bias_field, 0.8, 1.2)
        return bias_field

    def _estimate_gaussian_bias_field(self, img_data):
        smoothed = gaussian_filter(img_data, sigma=15)
        smoothed_mean = np.mean(smoothed)
        bias_field = smoothed / smoothed_mean
        bias_field = np.clip(bias_field, 0.7, 1.3)
        return bias_field

    def _estimate_polynomial_bias_field(self, img_data, degree=3):
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, img_data.shape[0]),
            np.linspace(-1, 1, img_data.shape[1]),
            np.linspace(-1, 1, img_data.shape[2]),
            indexing='ij'
        )
        coords = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        data_flat = img_data.flatten()
        threshold = np.percentile(data_flat[data_flat > 0], 10) if np.any(data_flat > 0) else 0
        mask = data_flat > threshold
        if np.sum(mask) < 100:
            print("Warning: Insufficient data points for bias field estimation, using uniform field")
            return np.ones_like(img_data)
        coords_masked = coords[mask]
        data_masked = data_flat[mask]
        if np.isnan(data_masked).any() or np.isinf(data_masked).any():
            print("Warning: Invalid data in bias field estimation, using uniform field")
            return np.ones_like(img_data)
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            poly = PolynomialFeatures(degree=degree)
            coords_poly = poly.fit_transform(coords_masked)
            model = LinearRegression()
            model.fit(coords_poly, data_masked)
            coords_poly_all = poly.transform(coords)
            bias_field_flat = model.predict(coords_poly_all)
            bias_field = bias_field_flat.reshape(img_data.shape)
            bias_field_mean = np.mean(bias_field)
            bias_field = bias_field / bias_field_mean
            bias_field = np.clip(bias_field, 0.5, 2.0)
            return bias_field
        except Exception as e:
            print(f"Warning: Bias field estimation failed: {str(e)}, using uniform field")
            return np.ones_like(img_data)

    def correct_bias_field(self, img_data, bias_field):
        print("Correcting bias field...")
        bias_field = np.maximum(bias_field, 0.1)
        corrected = img_data / bias_field
        original_range = np.percentile(img_data[img_data > 0], [1, 99])
        corrected = np.clip(corrected, original_range[0], original_range[1])
        if np.std(corrected) < np.std(img_data) * 0.1:
            print("Warning: Bias correction too aggressive, using original image")
            return img_data
        return corrected

    def visualize_bias_field(self, original, bias_field, corrected, output_path):
        print("Creating bias field visualization...")
        mid_slice = original.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im1 = axes[0].imshow(original[:, :, mid_slice], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        im2 = axes[1].imshow(bias_field[:, :, mid_slice], cmap='hot')
        axes[1].set_title('Estimated Bias Field')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        im3 = axes[2].imshow(corrected[:, :, mid_slice], cmap='gray')
        axes[2].set_title('Bias Field Corrected')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def normalize_intensity(self, img_data, method='zscore'):
        print("Normalizing intensity...")
        if method == 'zscore':
            mean_val = np.mean(img_data[img_data > 0])
            std_val = np.std(img_data[img_data > 0])
            normalized = (img_data - mean_val) / std_val
        elif method == 'minmax':
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            normalized = (img_data - min_val) / (max_val - min_val)
        elif method == 'histogram':
            from skimage import exposure
            normalized = exposure.equalize_hist(img_data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        return normalized

    def fast_gaussian_bias_correction(self, img_path):
        print("Applying fast Gaussian bias field correction...")
        img = nib.load(img_path)
        data = img.get_fdata()
        bias_field = gaussian_filter(data, sigma=20)
        bias_field_mean = np.mean(bias_field)
        bias_field = bias_field / bias_field_mean
        bias_field = np.clip(bias_field, 0.8, 1.2)
        corrected = data / bias_field
        corrected = np.clip(corrected, np.percentile(data[data > 0], 1),
                           np.percentile(data[data > 0], 99))
        return corrected, img.affine, img

    def improved_register_to_template(self, moving_img_sitk, template_img_sitk):
        print("Registering to MNI152 template using SimpleITK...")
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
            template_img_sitk, moving_img_sitk, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY))
        final_transform = registration_method.Execute(template_img_sitk, moving_img_sitk)
        resampled = sitk.Resample(moving_img_sitk, template_img_sitk, final_transform,
                                  sitk.sitkLinear, 0.0, moving_img_sitk.GetPixelID())
        return sitk.GetArrayFromImage(resampled), resampled

    def skull_stripping(self, img_data, method='robust', use_center_mask=True):
        print("Performing skull stripping...")
        if method == 'otsu':
            brain_mask = self._robust_skull_stripping(img_data)
        elif method == 'template' and self.template_mask_data is not None:
            brain_mask = self.template_mask_data > 0.5
        elif method == 'watershed':
            brain_mask = self._watershed_skull_stripping(img_data)
        elif method == 'robust':
            brain_mask = self._robust_skull_stripping(img_data)
        else:
            raise ValueError(f"Unknown skull stripping method: {method}")
        if use_center_mask and method != 'template':
            brain_mask = self._create_brain_centered_mask(img_data, brain_mask)
        stripped_data = img_data * brain_mask
        return stripped_data, brain_mask

    def _robust_skull_stripping(self, img_data):
        threshold = filters.threshold_otsu(img_data)
        initial_mask = img_data > threshold
        initial_mask = morphology.remove_small_objects(initial_mask, min_size=1000)
        try:
            initial_mask = morphology.binary_fill_holes(initial_mask)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                initial_mask = binary_fill_holes(initial_mask)
            except ImportError:
                from scipy.ndimage import binary_fill_holes
                initial_mask = binary_fill_holes(initial_mask)
        selem = morphology.ball(3)
        eroded_mask = morphology.binary_erosion(initial_mask, selem)
        labels = measure.label(eroded_mask)
        if labels.max() > 0:
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            brain_mask = largest_cc
        else:
            print("Warning: No connected components found after erosion, using original mask")
            brain_mask = initial_mask
        selem = morphology.ball(2)
        brain_mask = morphology.binary_dilation(brain_mask, selem)
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(3))
        brain_mask = morphology.binary_opening(brain_mask, morphology.ball(2))
        try:
            brain_mask = morphology.binary_fill_holes(brain_mask)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                brain_mask = binary_fill_holes(brain_mask)
            except ImportError:
                from scipy.ndimage import binary_fill_holes
                brain_mask = binary_fill_holes(brain_mask)
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=500)
        return brain_mask

    def _watershed_skull_stripping(self, img_data):
        from skimage.segmentation import watershed
        try:
            from skimage.feature import peak_local_maxima
        except ImportError:
            try:
                from skimage.feature.peak import peak_local_maxima
            except ImportError:
                try:
                    from skimage.feature import peak_local_maxima
                except ImportError:
                    print("Warning: peak_local_maxima not available, using fallback method")
                    return self._robust_skull_stripping(img_data)
        threshold = filters.threshold_otsu(img_data)
        binary = img_data > threshold
        binary = morphology.remove_small_objects(binary, min_size=1000)
        try:
            binary = morphology.binary_fill_holes(binary)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                binary = binary_fill_holes(binary)
            except ImportError:
                from scipy.ndimage import binary_fill_holes
                binary = binary_fill_holes(binary)
        distance = ndimage.distance_transform_edt(binary)
        try:
            try:
                local_maxi = peak_local_maxima(distance, labels=binary, 
                                             min_distance=15, exclude_border=False,
                                             threshold_abs=0.1)
            except TypeError:
                local_maxi = peak_local_maxima(distance, labels=binary, 
                                             min_distance=15, exclude_border=False)
            if len(local_maxi) > 0:
                local_maxi = np.ravel_multi_index(local_maxi.T, distance.shape)
                markers = np.zeros_like(distance, dtype=int)
                markers.ravel()[local_maxi] = range(len(local_maxi))
                brain_mask = watershed(-distance, markers, mask=binary)
                brain_mask = brain_mask > 0
                brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
                try:
                    brain_mask = morphology.binary_fill_holes(brain_mask)
                except AttributeError:
                    try:
                        from skimage.morphology import binary_fill_holes
                        brain_mask = binary_fill_holes(brain_mask)
                    except ImportError:
                        from scipy.ndimage import binary_fill_holes
                        brain_mask = binary_fill_holes(brain_mask)
            else:
                print("Warning: No local maxima found, using fallback method")
                brain_mask = self._robust_skull_stripping(img_data)
        except Exception as e:
            print(f"Warning: Watershed failed: {str(e)}, using fallback method")
            brain_mask = self._robust_skull_stripping(img_data)
        return brain_mask

    def _create_brain_centered_mask(self, img_data, brain_mask):
        brain_coords = np.where(brain_mask)
        if len(brain_coords[0]) == 0:
            return brain_mask
        center_x = np.mean(brain_coords[0])
        center_y = np.mean(brain_coords[1])
        center_z = np.mean(brain_coords[2])
        x, y, z = np.meshgrid(np.arange(img_data.shape[0]),
                             np.arange(img_data.shape[1]),
                             np.arange(img_data.shape[2]), indexing='ij')
        radius = min(img_data.shape) * 0.4
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        spherical_mask = distance <= radius
        refined_mask = brain_mask & spherical_mask
        return refined_mask

    def preprocess_single_image(self, input_file, output_prefix):
        print(f"\nProcessing: {input_file}")
        try:
            corrected_np, img_affine, orig_img = self.fast_gaussian_bias_correction(input_file)
            results = {'original': corrected_np, 'affine': img_affine}
            try:
                denoised = self.denoise_image(corrected_np)
                results['denoised'] = denoised
            except Exception as e:
                print(f"Warning: Denoising failed, using bias corrected data: {str(e)}")
                results['denoised'] = corrected_np
            try:
                normalized = self.normalize_intensity(results['denoised'])
                results['normalized'] = normalized
            except Exception as e:
                print(f"Warning: Intensity normalization failed, using denoised data: {str(e)}")
                results['normalized'] = results['denoised']
            norm = results['normalized']
            if norm.ndim == 4 and norm.shape[-1] == 1:
                norm = norm[..., 0]
            elif norm.ndim != 3:
                raise ValueError(f"Normalized image is not 3D: shape={norm.shape}")
            results['normalized'] = norm
            print(f"Normalized shape: {results['normalized'].shape}, Template shape: {self.template_data.shape}")
            template_sitk = sitk.ReadImage(self.template_path)
            moving_sitk = sitk.GetImageFromArray(results['normalized'])
            moving_sitk.SetSpacing(template_sitk.GetSpacing())
            moving_sitk.SetOrigin(template_sitk.GetOrigin())
            moving_sitk.SetDirection(template_sitk.GetDirection())
            moving_sitk = sitk.Cast(moving_sitk, sitk.sitkFloat32)
            template_sitk = sitk.Cast(template_sitk, sitk.sitkFloat32)
            try:
                registered_np, registered_sitk = self.improved_register_to_template(
                    moving_sitk, template_sitk)
                results['registered'] = registered_np
                results['registered_affine'] = np.eye(4)
            except Exception as e:
                print(f"Warning: Registration failed, using normalized data: {str(e)}")
                results['registered'] = results['normalized']
                results['registered_affine'] = np.eye(4)
            try:
                stripped, brain_mask = self.skull_stripping(results['registered'])
                results['stripped'] = stripped
                results['brain_mask'] = brain_mask
            except Exception as e:
                print(f"Warning: Skull stripping failed, using registered data: {str(e)}")
                results['stripped'] = results['registered']
                results['brain_mask'] = np.ones_like(results['registered'], dtype=bool)
            self.save_results(results, output_prefix)
            try:
                self.create_visualizations(results, output_prefix)
            except Exception as e:
                print(f"Warning: Visualization creation failed: {str(e)}")
            return results
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            raise

    def save_results(self, results, output_prefix):
        print("Saving results...")
        registered_affine = results.get('registered_affine', results.get('affine'))
        if 'stripped' in results:
            final_img = nib.Nifti1Image(results['stripped'], registered_affine)
            out_path = f"{output_prefix}_preprocessed.nii.gz"
            print(f"Saving preprocessed image to: {out_path}")
            nib.save(final_img, out_path)
            print(f"Saved preprocessed file: {out_path}")

    def create_visualizations(self, results, output_prefix):
        print("Creating visualizations...")
        self.visualize_bias_field(
            results['original'],
            results['bias_field'],
            results['bias_corrected'],
            os.path.join(self.output_dir, f"{output_prefix}_bias_field.png")
        )
        self.create_pipeline_overview(results, output_prefix)

    def create_pipeline_overview(self, results, output_prefix):
        mid_slice = results['original'].shape[2] // 2
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(results['original'][:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(results['denoised'][:, :, mid_slice], cmap='gray')
        axes[0, 1].set_title('Denoised')
        axes[0, 1].axis('off')
        axes[0, 2].imshow(results['bias_corrected'][:, :, mid_slice], cmap='gray')
        axes[0, 2].set_title('Bias Corrected')
        axes[0, 2].axis('off')
        axes[1, 0].imshow(results['normalized'][:, :, mid_slice], cmap='gray')
        axes[1, 0].set_title('Normalized')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(results['registered'][:, :, mid_slice], cmap='gray')
        axes[1, 1].set_title('Registered')
        axes[1, 1].axis('off')
        axes[1, 2].imshow(results['stripped'][:, :, mid_slice], cmap='gray')
        axes[1, 2].set_title('Skull Stripped')
        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_pipeline_overview.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def process_all_images(self):
        print("Starting parallel batch processing of all T1-weighted images...")
        os.makedirs(self.output_dir, exist_ok=True)
        input_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    input_files.append(os.path.join(root, file))
        if not input_files:
            print(f"No .nii.gz files found in {self.input_dir}")
            return
        print(f"Found {len(input_files)} files to process")
        args_list = []
        for f in input_files:
            rel_path = os.path.relpath(f, self.input_dir)
            rel_prefix = os.path.splitext(rel_path)[0]
            out_dir = os.path.join(self.output_dir, os.path.dirname(rel_prefix))
            os.makedirs(out_dir, exist_ok=True)
            output_prefix = os.path.join(out_dir, os.path.splitext(os.path.basename(rel_prefix))[0])
            preprocessed_file = f"{output_prefix}_preprocessed.nii.gz"
            if os.path.exists(preprocessed_file):
                print(f"Skipping {f} (already preprocessed)")
                continue
            pipeline_config = {
                'input_dir': self.input_dir,
                'template_dir': self.template_dir,
                'output_dir': self.output_dir
            }
            args_list.append((pipeline_config, f, rel_prefix))
        with ProcessPoolExecutor(max_workers=2) as executor:
            executor.map(process_image_wrapper, args_list)
        print(f"\nBatch processing completed. Results saved in: {self.output_dir}")

def main():
    pipeline = MRIPreprocessingPipeline()
    pipeline.process_all_images()

if __name__ == "__main__":
    main()
