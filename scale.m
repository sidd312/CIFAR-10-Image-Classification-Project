function [ scaled_data ] = scale( data )
    upper=max(data);
    lower=min(data);
    N=size(data,1);
    scaled_data=(data-repmat(lower,N,1))./repmat(upper-lower,N,1);
end

