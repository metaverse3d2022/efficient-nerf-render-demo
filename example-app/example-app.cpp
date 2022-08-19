#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <assert.h>

#define PI 3.141592653589793

using namespace torch::indexing;

torch::Tensor trans_t(double t) {
  return torch::tensor({{1., 0., 0., 0.}, \
                        {0., 1., 0., 0.}, \
                        {0., 0., 1., t }, \
                        {0., 0., 0., 1.}});
}

torch::Tensor rot_phi(double phi) {
  return torch::tensor({{1., 0., 0., 0.}, \
                        {0., std::cos(phi),-std::sin(phi), 0.}, \
                        {0., std::sin(phi), std::cos(phi), 0.}, \
                        {0., 0., 0., 1.}});
}

torch::Tensor rot_theta(double th) {
  return torch::tensor({{std::cos(th),0.,-std::sin(th), 0.}, \
                        {0., 1., 0., 0.}, \
                        {std::sin(th),0., std::cos(th), 0.}, \
                        {0., 0., 0., 1.}});
}

torch::Tensor sigma2weights(torch::Tensor deltas, torch::Tensor sigmas) {
  auto alphas = 1.0 - torch::exp(-deltas*torch::softplus(sigmas));
  auto alphas_shifted = torch::cat({torch::ones_like(alphas.index({Slice(), Slice(None,1)})), 1.0-alphas+1e-10}, -1);
  torch::Tensor weights = alphas * torch::cumprod(alphas_shifted, -1).index({Slice(), Slice(None, -1)});
  return weights;
}

torch::Tensor eval_sh(int deg, torch::Tensor sh, torch::Tensor dirs) {
    /*
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]

    Returns:
        [..., C]
    */
    double C0 = 0.28209479177387814;
    double C1 = 0.4886025119029199;
    double C2[5] = {1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396};
    
    assert(deg <= 2 && deg >= 0);
    assert((deg + 1) * (deg + 1) == sh.size(-1));
    int C = sh.size(-2);
    auto result = C0 * sh.index({"...", 0});
    
    if (deg > 0) {
        auto x = dirs.index({"...", Slice(0, None, 1)});
        auto y = dirs.index({"...", Slice(1, None, 2)});
        auto z = dirs.index({"...", Slice(2, None, 3)});
        result = (result -                         \
                C1 * y * sh.index({"...", 1}) +    \
                C1 * z * sh.index({"...", 2}) -    \
                C1 * x * sh.index({"...", 3}));
        if (deg > 1) {
            auto xx = x * x;
            auto yy = y * y;
            auto zz = z * z;
            auto xy = x * y;
            auto yz = y * z;
            auto xz = x * z;
            result = (result +                                             \
                    C2[0] * xy * sh.index({"...", 4}) +                    \
                    C2[1] * yz * sh.index({"...", 5}) +                    \
                    C2[2] * (2.0 * zz - xx - yy) * sh.index({"...", 6}) +  \
                    C2[3] * xz * sh.index({"...", 7}) +                    \
                    C2[4] * (xx - yy) * sh.index({"...", 8}));
        }
    }
    
    return result;
}

torch::Tensor pose_spherical(double theta, double phi, double radius) {
  torch::Tensor c2w = trans_t(radius);
  c2w = torch::matmul(rot_phi(phi/180.*PI), c2w);
  c2w = torch::matmul(rot_theta(theta/180.*PI), c2w);
  c2w = torch::matmul(torch::tensor({{-1., 0., 0., 0.}, {0., 0., 1., 0.}, \
                      {0., 1., 0., 0.}, {0., 0., 0., 1.}}), c2w);
  return c2w;
}

torch::Tensor get_ray_directions(int H, int W, double focal) {
  auto cood = torch::meshgrid({torch::linspace(0, W-1, W), torch::linspace(0, H-1, H)});
  auto i = cood[0].transpose(1,0);
  auto j = cood[1].transpose(1,0);
  torch::Tensor dirs = torch::stack({(i-W/2.)/focal, -(j-H/2.)/focal, -torch::ones_like(i)}, -1); // (H, W, 3)
  return dirs;
}

std::vector<torch::Tensor> get_rays(torch::Tensor directions, torch::Tensor c2w) {

    int H = directions.size(0);
    int W = directions.size(1);
    auto rays_d = torch::empty_like(directions);
    
    for(int w=0; w<W; w++)
        for(int h=0; h<H; h++) {
            rays_d.index_put_({h, w}, torch::matmul(c2w.index({Slice(None,3), Slice(None,3)}), directions.index({h, w})));
        }
    
    auto rays_o = c2w.index({Slice(None,3), 3}).expand_as(rays_d);

    rays_o = rays_o.reshape({-1,3});
    rays_d = rays_d.reshape({-1,3});
    
    std::vector<torch::Tensor> rays{rays_o, rays_d};

    return rays;
}

torch::Tensor calc_index_coarse(torch::Tensor xyz) {
  double coord_scope = 3.0;
  int grid_coarse = 384;
  double xyz_min = -coord_scope;
  double xyz_max = coord_scope;
  double xyz_scope = xyz_max - xyz_min;
  
  auto ijk_coarse = ((xyz - xyz_min) / xyz_scope * grid_coarse).to(torch::kLong).clamp(0, grid_coarse-1);
  return ijk_coarse;
}

torch::Tensor calc_index_fine(torch::Tensor xyz) {
  double coord_scope = 3.0;
  int grid_coarse = 384;
  int grid_fine = 3;
  int res_fine = grid_coarse * grid_fine;
  double xyz_min = -coord_scope;
  double xyz_max = coord_scope;
  double xyz_scope = xyz_max - xyz_min;
  
  auto xyz_norm = (xyz - xyz_min) / xyz_scope;
  auto xyz_fine = (xyz_norm * res_fine).to(torch::kLong);
  auto index_fine = xyz_fine % grid_fine;
  
  return index_fine;
}

torch::Tensor query_coarse_sigma(torch::Tensor xyz) {
  torch::jit::script::Module voxels_dict = torch::jit::load("voxels_dict.pt");
  torch::Tensor sigma_voxels_coarse = voxels_dict.attr("sigma_voxels_coarse").toTensor();
  torch::Tensor ijk_coarse = calc_index_coarse(xyz);
  torch::Tensor out = sigma_voxels_coarse.index({ijk_coarse.index({Slice(), 0}), ijk_coarse.index({Slice(), 1}), ijk_coarse.index({Slice(), 2})});
  return out;
}

torch::Tensor query_coarse_index(torch::Tensor xyz) {
  torch::jit::script::Module voxels_dict = torch::jit::load("voxels_dict.pt");
  torch::Tensor index_voxels_coarse = voxels_dict.attr("index_voxels_coarse").toTensor();
  torch::Tensor ijk_coarse = calc_index_coarse(xyz);
  torch::Tensor out = index_voxels_coarse.index({ijk_coarse.index({Slice(), 0}), ijk_coarse.index({Slice(), 1}), ijk_coarse.index({Slice(), 2})});
  return out;
}

torch::Tensor query_fine(torch::Tensor xyz) {
  torch::jit::script::Module voxels_dict = torch::jit::load("voxels_dict.pt");
  torch::Tensor voxels_fine = voxels_dict.attr("voxels_fine").toTensor();
  // calc index_coarse
  torch::Tensor index_coarse = query_coarse_index(xyz);
  // calc index_fine
  torch::Tensor ijk_fine = calc_index_fine(xyz);
  torch::Tensor out = voxels_fine.index({index_coarse, ijk_fine.index({Slice(), 0}), ijk_fine.index({Slice(), 1}), ijk_fine.index({Slice(), 2})});
  return out;
}

torch::Tensor inference(torch::Tensor xyz_, torch::Tensor dir_, torch::Tensor z_vals, torch::Tensor idx_render) {
  int N_rays = xyz_.size(0);
  int N_samples_ = xyz_.size(1);
  // Embed directions
  xyz_ = xyz_.index({idx_render.index({Slice(), 0}), idx_render.index({Slice(), 1})}).view({-1, 3});
  auto view_dir = dir_.unsqueeze(1).expand({-1, N_samples_, -1});
  view_dir = view_dir.index({idx_render.index({Slice(), 0}), idx_render.index({Slice(), 1})});
  
  int deg = 2;
  int dim_sh = 3 * (deg + 1) * (deg + 1);
  
  auto out_fine = query_fine(xyz_);
  auto outputs = torch::split(out_fine, {1, dim_sh}, -1);
  auto sigma = outputs[0];
  auto sh = outputs[1];
  
  auto rgb = eval_sh(deg, sh.reshape({-1, 3, (deg + 1) * (deg + 1)}), view_dir);
  rgb = torch::sigmoid(rgb);
  
  double sigma_default = -20.0;
  auto rgbs = torch::full({N_rays, N_samples_, 3}, 1.0);
  auto sigmas = torch::full({N_rays, N_samples_, 1}, sigma_default);
  rgbs.index_put_({idx_render.index({Slice(), 0}), idx_render.index({Slice(), 1})}, rgb);
  sigmas.index_put_({idx_render.index({Slice(), 0}), idx_render.index({Slice(), 1})}, sigma);
  sigmas = sigmas.squeeze(-1);
  
  // Convert these values using volume rendering (Section 4)
  auto deltas = z_vals.index({Slice(), Slice(1, None)}) - z_vals.index({Slice(), Slice(None, -1)}); // (N_rays, N_samples_-1)
  auto delta_inf = 1e5 * torch::ones_like(deltas.index({Slice(), Slice(None, 1)})); // (N_rays, 1) the last delta is infinity
  deltas = torch::cat({deltas, delta_inf}, -1); // (N_rays, N_samples_)
  auto weights = sigma2weights(deltas, sigmas);
  auto weights_sum = weights.sum(1); // (N_rays)
  
  auto rgb_final = torch::sum(weights.unsqueeze(-1)*rgbs, -2); // (N_rays, 3)

  // white_back = true
  rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1);
  
  return rgb_final;
}

torch::Tensor render_rays(torch::Tensor ray_batch, int N_samples=128, int N_importance=5, float perturb=0.) {
  int N_rays = ray_batch.size(0);
  auto rays_o = ray_batch.index({Slice(), Slice(0, 3)});
  auto rays_d = ray_batch.index({Slice(), Slice(3, 6)});
    
    
  // z_vals_coarse
  //int N_samples = 128;
  int N_samples_coarse = N_samples;
  auto z_vals_coarse = torch::linspace(0., 1., N_samples_coarse);
  //use linear sampling in depth space
  double near = 2.0; // val_dataset.near
  double far = 6.0; // val_dataset.far
  z_vals_coarse = near * (1-z_vals_coarse) + far * z_vals_coarse;
  z_vals_coarse = z_vals_coarse.unsqueeze(0);
  
  
  // z_vals_fine
  //int N_importance = 5;
  int N_samples_fine = N_samples * N_importance;
  auto z_vals_fine = torch::linspace(0., 1., N_samples_fine);
  //use linear sampling in depth space
  z_vals_fine = near * (1-z_vals_fine) + far * z_vals_fine;
  z_vals_fine = z_vals_fine.unsqueeze(0);
  

  z_vals_coarse = z_vals_coarse.expand({N_rays, -1});
  
  auto xyz_sampled_coarse = rays_o.unsqueeze(1) + \
                              rays_d.unsqueeze(1) * z_vals_coarse.unsqueeze(2); // (N_rays, N_samples_coarse, 3)
  
  auto xyz_coarse = xyz_sampled_coarse.reshape({-1, 3});

  auto sigmas = query_coarse_sigma(xyz_coarse).reshape({N_rays, N_samples_coarse});
  
  auto deltas_coarse = z_vals_coarse.index({Slice(), Slice(1, None)}) - z_vals_coarse.index({Slice(), Slice(None, -1)});
  auto delta_inf = 1e5 * torch::ones_like(deltas_coarse.index({Slice(), Slice(None, 1)}));
  deltas_coarse = torch::cat({deltas_coarse, delta_inf}, -1);
  auto weights_coarse = sigma2weights(deltas_coarse, sigmas);
  
  double weight_threashold = 1e-5;
  auto idx_render = torch::nonzero(weights_coarse >= std::min(weight_threashold, weights_coarse.max().item().to<double>()));
  //double t =  weights_coarse.max().item().to<double>();
  
  int scale = 5;
  idx_render = idx_render.unsqueeze(1).expand({-1, scale, -1});
  auto idx_render_fine = idx_render.clone();
  idx_render_fine.index_put_({"...", 1}, idx_render.index({"...", 1}) * scale + torch::arange(scale).reshape({1, scale}));
  idx_render_fine = idx_render_fine.reshape({-1, 2});
  auto xyz_sampled_fine = rays_o.unsqueeze(1) + \
                              rays_d.unsqueeze(1) * z_vals_fine.unsqueeze(2); // (N_rays, N_samples*scale, 3)
  
  torch::Tensor rgb_fine = inference(xyz_sampled_fine, rays_d, z_vals_fine, idx_render_fine);
  
  //std::cout << z_vals_coarse.sizes() << std::endl;
  //std::cout << z_vals_fine.sizes() << std::endl;
  
  
  
  return rgb_fine;
  
}

int main() {
  auto c2w = pose_spherical(90, -30, 4);
  int w = 200;
  int h = 200;
  double focal = 0.5 * w / std::tan(0.5*0.6911112070083618);
  auto directions = get_ray_directions(h, w, focal);
  //std::cout << directions << std::endl;
  directions = directions / torch::norm(directions, 2, -1, true);
  auto _rays = get_rays(directions, c2w);
  auto rays_o = _rays[0];
  auto rays_d = _rays[1];
  auto rays = torch::cat({rays_o, rays_d}, 1);
  //std::cout << rays << std::endl;
  
  auto result = render_rays(rays.index({Slice(), Slice()}), 128);
  std::cout << result.sizes() << std::endl;
  
  auto rgb_map = result.view({h, w, 3});
  
  //std::cout << rgb_map[50] << std::endl;
  
  torch::save({rgb_map}, "tensors.pt");

  
}