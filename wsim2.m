function [WSIM2]=wsim2(B, idx_test, Theta, Phi, is_symmetric)

    [N, K] = size(Theta);

    [max_v, c] = max(Theta');
    theta_hard = zeros(size(Theta));
    lidx = sub2ind(size(Theta), 1:N, c);            % Convert To Linear Indexing
    theta_hard(lidx) = 1;
    c_len = sum(theta_hard);

    norm = c_len' * c_len;

    if is_symmetric
        norm(find(eye(size(norm)))) = (diag(norm)' - c_len)/2;
    else
        norm(find(eye(size(norm)))) = diag(norm)' - c_len;
    end
    % mask <=0 values !
    norm(norm<=0) = 1;
    
    pp = zeros([K,K]);
    [is,js]=ind2sub(size(B), idx_test);

    for i=1:length(is)
        i_ = is(i);
        j_ = js(i);
        w_ = full(B(i_, j_));
        pp(c(i_), c(j_)) = pp(c(i_), c(j_)) + w_;
    end

    pp = pp / norm;

    wd = zeros(1,length(idx_test));
    ws = zeros(1, length(idx_test));
    for i=1:length(idx_test)
        i_ = is(i);
        j_ = js(i);

        %fprintf('ci %d, cj %d, pp_ij %.2f, b_ij %.2f\n', c(i_),c(j_),pp(c(i_), c(j_)), full(B(i_, j_)) );
        ws(i) = pp(c(i_), c(j_));
        wd(i) = full(B(i_, j_));
    end

    WSIM2 = mean((ws - wd).^2); % MSE
