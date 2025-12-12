function M = assembleReaction(mesh, feMap)
    
    % Local reference reaction matrix
    Mref = 1/12*[1 1/2 1/2; 1/2 1 1/2; 1/2 1/2 1];
    
    % Node indices for assembly
    nodeIndI = [1 2 3 1 2 3 1 2 3];
    nodeIndJ = [1 1 1 2 2 2 3 3 3];
    
    % Global row and column indices for sparse matrix assembly
    globRows = mesh.meshElements(nodeIndI, :);
    globCols = mesh.meshElements(nodeIndJ, :);
    
    % Assemble global reaction matrix in vector form
    MVector = repmat(Mref(:),[1 mesh.numMeshElements]) .* repmat(feMap.J,[9 1]);
    
    % Assemble the global reaction matrix using sparse format
    M = sparse(globRows(:), globCols(:), MVector(:), mesh.numVertices, mesh.numVertices);

end