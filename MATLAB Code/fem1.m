% Number of timesteps
m = 350;
% Type of different diffusivity
index = 1;
% Current mesh
meshname = 'Meshes/mesh_0256.msh';

% Final time value
T = 35;

% Define f
a = 18.515;
fr = 0;
ft = 0.2383;
fd = 1;
f = @(x) a.*(x-fr).*(x-ft).*(x-fd);

% Import and define mesh and finite element map
mesh = Mesh2D(meshname);
FEmap = FEMap(mesh);

% Set the diffusivity value
Sigma_val = 9.5298e-4;
Sigma_mod = [10*Sigma_val, Sigma_val, 0.1*Sigma_val];
Sigma = repmat(Sigma_val, [1 mesh.numMeshElements]);
Sigma(mesh.meshElementFlags ~= 3) = Sigma_mod(index);

% Initialize the matrix with the m+1 solutions
u = zeros(m+1, mesh.numVertices);
% Set initial condition
u(1,:) = (mesh.vertices(1,:) <= 0.1 & mesh.vertices(2,:) >= 0.9);
% Define diffusion matrix
A = assembleDiffusion(mesh, FEmap, Sigma);
% Define reaction matrix
M = assembleReaction(mesh, FEmap);
%M = diag(sum(M,2));

% Calculate timestep
dt = T/m;

% Calculate left-hand side matrix
LHS = dt.*A + M;

video = VideoWriter("solution_FEM.avi");
video.FrameRate = 2;
open(video)
for i = 2:m+1
    % Iteratively solve system at each timestep
    u_prev = u(i-1,:)';
    rhs = M*u_prev - dt.*M*f(u_prev);
    u(i,:) = (LHS\rhs)';
    
    subplot(1,2,1)
    % Plot solution and save video
    if (mod(i,20)==0)
        mesh.plotSolution(u(i,:)');
        axis equal
        title(num2str(i))
        subplot(1,2,2)
        mesh.plotSolution(u(i,:)');
        zlim([0 1])
        axis equal
        view(2)
        frame = getframe(gcf);
        writeVideo(video,frame)
        pause(0.1)
    end
end
close(video)

% Check the activation time
act_time = find(sum(ft-u>1e-6,2),1,'last')*dt;
% Check if the potential remains between 0 and 1
potential_exceeds = (sum(((u < 0) | (u > 1)),'all') > 0);
potential_exceeds_2 = (sum(((u < -1e-10) | (u > 1 + 1e-10)),'all') > 0);
% Check if the matrix is an M-matrix
[L,U] = lu(LHS);
isM = (sum(diag(L) <= eps) == 0) && (sum(diag(U) <= eps) == 0);
% isM = prod(diag(LHS) > 0, 'all');
% isM = isM * prod(diag(LHS) >-sum(LHS-diag(diag(LHS)),2), 'all');
for i = 1:length(LHS(1,:))
    temp = LHS(i,:);
    temp(i) = 0;
    isM = isM * prod(temp <= eps);
    if isM == 0
        break
    end
end