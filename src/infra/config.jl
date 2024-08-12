"""
    ColBERTConfig(; use_gpu::Bool, rank::Int, nranks::Int, query_token_id::String,
            doc_token_id::String, query_token::String, doc_token::String, checkpoint::String,
            collection::String, dim::Int, doc_maxlen::Int, mask_punctuation::Bool,
            query_maxlen::Int, attend_to_mask_tokens::Bool, index_path::String,
            index_bsize::Int, nbits::Int, kmeans_niters::Int, nprobe::Int, ncandidates::Int)

Structure containing config for running and training various components.

# Arguments

  - `use_gpu`: Whether to use a GPU or not. Default is `false`.
  - `rank`: The index of the running GPU. Default is `0`. For now, the package only allows this to be `0`.
  - `nranks`: The number of GPUs used in the run. Default is `1`. For now, the package only supports one GPU.
  - `query_token_id`: Unique identifier for query tokens (defaults to `[unused0]`).
  - `doc_token_id`: Unique identifier for document tokens (defaults to `[unused1]`).
  - `query_token`: Token used to represent a query token (defaults to `[Q]`).
  - `doc_token`: Token used to represent a document token (defaults to `[D]`).
  - `checkpoint`: The path to the HuggingFace checkpoint of the underlying ColBERT model. Defaults to `"colbert-ir/colbertv2.0"`.
  - `collection`: Path to the file containing the documents. Default is `""`.
  - `dim`: The dimension of the document embedding space. Default is 128.
  - `doc_maxlen`: The maximum length of a document before it is trimmed to fit. Default is 220.
  - `mask_punctuation`: Whether or not to mask punctuation characters tokens in the document. Default is true.
  - `query_maxlen`: The maximum length of queries after which they are trimmed.
  - `attend_to_mask_tokens`: Whether or not to attend to mask tokens in the query. Default value is false.
  - `index_path`: Path to save the index files.
  - `index_bsize`: Batch size used for some parts of indexing.
  - `chunksize`: Custom size of a chunk, i.e the number of passages for which data is to be stored in one chunk. Default is `missing`,
        in which case `chunksize` is determined from the size of the `collection` and `nranks`.
  - `nbits`: Number of bits used to compress residuals.
  - `kmeans_niters`: Number of iterations used for k-means clustering.
  - `nprobe`: The number of nearest centroids to fetch during a search. Default is `2`. Also see [`retrieve`](@ref).
  - `ncandidates`: The number of candidates to get during candidate generation in search. Default is `8192`. Also see [`retrieve`](@ref).

# Returns

A [`ColBERTConfig`](@ref) object.

# Examples

Most users will just want to use the defaults for most settings. Here's a minimal example:

```jldoctest
julia> using ColBERT;

julia> config = ColBERTConfig(
           use_gpu = true,
           collection = "/home/codetalker7/documents",
           index_path = "./local_index"
       );

```
"""
Base.@kwdef struct ColBERTConfig
    # run settings
    use_gpu::Bool = false
    rank::Int = 0
    nranks::Int = 1

    # tokenization settings
    query_token_id::String = "[unused0]"
    doc_token_id::String = "[unused1]"
    query_token::String = "[Q]"
    doc_token::String = "[D]"

    # resource settings 
    checkpoint::String = "colbert-ir/colbertv2.0"
    collection::String = ""

    # doc settings
    dim::Int = 128
    doc_maxlen::Int = 220
    mask_punctuation::Bool = true

    # query settings
    query_maxlen::Int = 32
    attend_to_mask_tokens::Bool = false

    # indexing settings
    index_path::String = ""
    index_bsize::Int = 64
    chunksize::Union{Missing, Int} = missing
    nbits::Int = 2
    kmeans_niters::Int = 20

    # search settings
    nprobe::Int = 2
    ncandidates::Int = 8192
end

"""
    save(config::ColBERTConfig)

Save a [`ColBERTConfig`](@ref) to disk in JSON.

# Arguments

  - `config`: The [`ColBERTConfig`](@ref) to save.

# Examples

```jldoctest
julia> using ColBERT;

julia> config = ColBERTConfig(
           use_gpu = true,
           collection = "/home/codetalker7/documents",
           index_path = "./local_index"
       );

julia> ColBERT.save(config);

```
"""
function save(config::ColBERTConfig)
    properties = [Pair{String, Any}(string(field), getproperty(config, field))
                  for field in fieldnames(ColBERTConfig)]
    isdir(config.index_path) || mkdir(config.index_path)
    open(joinpath(config.index_path, "config.json"), "w+") do io
        JSON.print(
            io,
            Dict(properties),
            4
        )
    end
end

"""
    load_config(index_path::String)

Load a [`ColBERTConfig`](@ref) from disk.

# Arguments

  - `index_path`: The path of the directory where the config resides.

# Examples

```jldoctest
julia> using ColBERT;

julia> config = ColBERTConfig(
           use_gpu = true,
           collection = "/home/codetalker7/documents",
           index_path = "./local_index"
       );

julia> ColBERT.save(config);

julia> ColBERT.load_config("./local_index")
ColBERTConfig(true, 0, 1, "[unused0]", "[unused1]", "[Q]", "[D]", "colbert-ir/colbertv2.0", "/home/codetalker7/documents", 128, 220, true, 32, false, "./local_index", 64, 2, 20, 2, 8192)
```
"""
function load_config(index_path::String)
    config_dict = JSON.parsefile(joinpath(index_path, "config.json"))
    key_vals = collect(zip(Symbol.(keys(config_dict)), values(config_dict)))
    eval(:(ColBERTConfig($([Expr(:kw, :($key), :($val)) for (key, val) in key_vals]...))))
end
