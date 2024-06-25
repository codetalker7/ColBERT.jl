# TODO: implement on-disk collections, and the case where pids are not necessarily sorted and can be arbitrary
"""
    Collection(path::String)

A wrapper around a collection of documents, which stores the underlying collection as a `Vector{String}`.

# Arguments

- `path::String`:   A path to the document dataset. It is assumed that `path` refers to a CSV file. Each line of the
                    the CSV file should be of the form `pid \\t document`, where `pid` is the integer index of the document. `pid`s should be in the range ``[1, N]``, where ``N`` is the number of documents, and should be sorted.

# Examples

Here's an example which loads a small subset of the LoTTe dataset defined in `short_collections.tsv` (see the `examples` folder in the package).

```julia-repl
julia> using ColBERT;

julia> dataroot = "downloads/lotte";

julia> dataset = "lifestyle";

julia> datasplit = "dev";

julia> path = joinpath(dataroot, dataset, datasplit, "short_collection.tsv")
"downloads/lotte/lifestyle/dev/short_collection.tsv"

julia> collection = Collection(path)
Collection at downloads/lotte/lifestyle/dev/short_collection.tsv with 10 passages.
```
"""
struct Collection
    path::String
    data::Vector{String}
end

function Collection(path::String)
    file = CSV.File(path; delim = '\t', header = [:pid, :text],
        types = Dict(:pid => Int, :text => String), debug = true, quoted = false)
    @info "Loaded $(length(file.text)[1]) passages."
    Collection(path, file.text)
end

"""
    get_chunksize(collection::Collection, nranks::Int)

Determine the size of chunks used to store the index, based on the size of the `collection` and the number of available GPUs.

# Arguments

- `collection::Collection`: The underlying collection of documents. 
- `nranks::Int`: Number of available GPUs to compute the index. At this point, the package only supports `nranks = 1`.

# Examples

Continuing from the example from the [`Collection`](@ref) constructor:

```julia-repl
julia> get_chunksize(collection, 1) 
11
```
"""
function get_chunksize(collection::Collection, nranks::Int)
    Int(min(25000, 1 + floor(length(collection.data) / nranks)))
end

"""
    enumerate_batches(collection::Collection; [chunksize, nranks])

Batch the `collection` into chunks containing tuples of the form `(chunk_idx, offset, passages)`, where `chunk_idx` is the index of the chunk, `offset` is the index of the first passsage in the chunk, and `passages` is a `Vector{String}` containing the passages in the chunk.

# Arguments

- `collection::Collection`: The collection to batch.
- `chunksize::Union{Int, Missing}`: The chunksize to use to batch the collection. Default `missing`. If this is `missing`, then `chunksize` is determined using [`get_chunksize`](@ref) based on the `collection` and `nranks`.
- `nranks::Union{Int, Missing}`: The number of available GPUs. Default `missing`. Currently the package only supports `nranks = 1`.

The `collection` is batched into chunks of uniform size (with the last chunk potentially having a smaller size).

# Examples

Continuing from the example in the [`Collection`](@ref) constructor.

```julia-repl
julia> enumerate_batches(collection; nranks = 1);

julia> enumerate_batches(collection; chunksize = 3); 
```
"""
function enumerate_batches(
        collection::Collection; chunksize::Union{Int, Missing} = missing,
        nranks::Union{Int, Missing} = missing)
    if ismissing(chunksize)
        if ismissing(nranks)
            error("Atleast one of the arguments chunksize or nranks must be specified!")
        end
        chunksize = get_chunksize(collection, nranks)
    end

    num_passages = length(collection.data)
    batches = Vector{Tuple{Int, Int, Vector{String}}}()
    chunk_idx, offset = 1, 1
    while true
        push!(batches,
            (chunk_idx, offset,
                collection.data[offset:min(offset + chunksize - 1, num_passages)]))
        chunk_idx += 1
        offset += chunksize

        if offset > num_passages
            break
        end
    end
    batches
end

function Base.show(io::IO, collection::Collection)
    print(io, "Collection at $(collection.path) with $(length(collection.data)) passages.")
end
