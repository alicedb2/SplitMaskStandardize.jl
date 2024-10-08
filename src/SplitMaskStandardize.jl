 module SplitMaskStandardize

    using DataFrames
    using CSV
    using StatsBase: std, mean
    using Random: shuffle!, default_rng, MersenneTwister, AbstractRNG

    export SMSDataset

    export mask, idx, filter, val
    export training, validation, test, nth
    export presence, presmask, presidx
    export absence, absmask, absidx
    export standardize, split


    const _reserved = [:training, :validation, :test,
                       :idx, :mask, :filter, :split,
                       :presence, :absence, :standardize,
                       :presidx, :absidx, :presmask, :absmask]

    struct SMSDataset
        __df::AbstractDataFrame
        __slices::Union{Nothing, Vector{UnitRange{Int64}}, Vector{Vector{Int64}}}
        __zero::AbstractDataFrame
        __scale::AbstractDataFrame
        __rng::AbstractRNG
    end

    """
    SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
    SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\\t", shuffle=true, subsample=nothing)
    SMSDataset(dataset::SMSDataset; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)

    Create a dataset object from a DataFrame or a CSV file.

    The dataset object is a barebone wrapper over a DataFrame which
    allows to quickly generate training, validation and test splits,
    extract presence and absence data, and standardize the data.
    Both object-oriented-style and piping syntax are supported.

    # Arguments:
    - `df`: DataFrame object.
    - `csvfile`: Path to a CSV file.
    - `splits`: Fractions (will be normalized) of the dataset to split into training, validation, and test sets
    - `delim`: Delimiter for the CSV file.
    - `shuffle`: Shuffle the dataset before splitting.
    - `subsample`: Number (integer) or fraction (float) of rows to subsample from the dataset.
    - `returncopy`: Return a copy of the DataFrame object.

    ## Splits
    - Splits are given as a vector of any numbers which are normalized to fractions.
      For example [2, 1, 2] will result in [0.4, 0.2, 0.4] splits.
    - When 3 splits are specified, the dataset is split into training, validation, and test sets.
    - When 2 splits are specified the dataset is split into training and test sets.
    - When an arbitrary number of splits are specified, the first
      and last split are considered as the training and test sets.
      The splits can be access using an index, i.e. dataset[i]

    ## Reserved properties/fields
    Properties/fields of the underlying DataFrames are exposed that are not
    "idx", "mask", "filter", "presence", "absence", "presmask" "absmask", "presidx", "absidx", "standardize", "training", "validation", "test", "split", or internal fields "\\_\\_df", "\\_\\_slices", "\\_\\_zero", "\\_\\_scale".

    # Basic examples
    ```julia-repl
    dataset = SMSDataset("data.csv")
    dataset = SMSDataset(dataframe)

    trainingset = dataset.training
    validationset = dataset.validation
    testset = dataset.test

    nth_split = dataset[n]
    nth_split = dataset |> nth(n)

    allsplits = [split for split in dataset]

    julia> dataset(:col)             # Return a column from the underlying DataFrame as a vector
    39024-element Vector{Float64}:
     25.518239974975586
     10.868968963623047
      ⋮
      2.759552240371704

    # This is equivalent to dataset |> val(:col)

    julia> dataset(:col1, :col2)
    2×39024 Matrix{Float64}:
     25.5182  10.869   …   2.75955
     11.3294  11.3649     12.8618


    julia> dataset.presence(:sp1)   # :sp1 must contain true/false/1/0 values
                                    # true/1 are considered as a presence
    SMSDataset(33129×81 DataFrame
       Row │ lon        lat      sp1      sp2      sp3      sp4      sp5     ⋯
           │ Float64    Float64  Float64  Float64  Float64  Float64  Float64 ⋯
    ───────┼──────────────────────────────────────────────────────────────────
         1 │  -84.25    33.4167      1.0      1.0      1.0      1.0      1.0 ⋯
         2 │  -95.25    35.75        1.0      1.0      1.0      0.0      1.0
       ⋮   │     ⋮         ⋮        ⋮        ⋮        ⋮        ⋮        ⋮    ⋱
     33129 │  -97.25    46.9167      1.0      1.0      1.0      1.0      1.0


    # presence(:col) and absence(:col) mask by casting Bool and !Bool on
    # the elements of :col, but a custom mask can be provided using filter

    julia> dataset.filter(:sp1, x -> x > 10)   # or dataset |> filter(:sp1, >10)
    SMSDataset(39024×81 DataFrame
       Row │ lon        lat      sp1      sp2      sp3      sp4      sp5     ⋯
           │ Float64    Float64  Float64  Float64  Float64  Float64  Float64 ⋯
    ───────┼──────────────────────────────────────────────────────────────────
         1 │  -84.25    33.4167      1.0      1.0      1.0      1.0      1.0 ⋯
         2 │  -95.25    35.75        1.0      1.0      1.0      0.0      1.0
       ⋮   │     ⋮         ⋮        ⋮        ⋮        ⋮        ⋮        ⋮    ⋱
     39024 │ -108.25    28.0833      0.0      0.0      0.0      0.0      0.0
    ```

    # Examples of chaining
    ```julia-repl
    dataset.absence(:species)      # :species must contain true/false/1/0 values
    dataset |> absence(:species)   # false/0 are considered as an absence

    dataset.absence(:species)(:col1, :col2, :col3, :col4)               # return a 4xN matrix of predictors associated
    dataset |> absence(:species) |> val(:col1, :col2, :col3, :col4)     # with absences of :species

    dataset.standardize(:col1, :col2, :col3)      # Return a copy of the dataset where col1, col2, and col3
    dataset |> standardize(:col1, :col2, :col3)   # have been standardized against the training set

    dataset.standardize(:)          # Standardize all numeric columns
    dataset |> standardize(:)                     

    dataset.standarize(:col1, :col2)(:col1, :col2, :col3)               # Return a 3xN matrix of stacked columns across all splits
    dataset |> standarize(:col1, :col2) |> val(:col1, :col2, :col3)     # where col1 and col2 have been standardized against the training set

    dataset.validation.presence(:species).standardize(:col1, :col2)     # Return dataset containing a copy of the underlying
    dataset.validation.absence(:species).standardize(:col1, :col2)      # dataframe at presences or absences of :species in the
                                                                        # validation set where col1 and col2 have been
                                                                        # standardized against the training set


    dataset |> validation() |> presence(:species) |> standardize(:col1, :col2) # Equivalent to the above
    dataset |> validation() |> absence(:species) |> standardize(:col1, :col2)

    dataset.training.presence(:species1).presence(:species2)            # Return simultaneous presences of both species in training set
    dataset |> training() |> presence(:species1) |> presence(:species2)

    dataset.test.presence(:species1).absence(:species1)                 # Return empty dataset
    dataset |> test() |> presence(:species1) |> absence(:species1)

    dataset.idx(:col, ismissing)                  # Return indices of missing values in :col
    dataset |> idx(:col, ismissing)

    dataset.test.presidx(:species)                # Return indices of presences of :species in the test set
    dataset |> test() |> presidx(:species)

    dataset.training.absidx(:species)             # Return indices of absences of :species in the training set
    dataset |> training() |> absidx(:species)

    dataset.presmask(:species)                    # Return a mask of presences of :species in the dataset
    dataset |> presmask(:species)

    dataset.absmask(:species)                     # Return a mask of absences of :species in the dataset
    dataset |> absmask(:species)

    dataset.training.mask(:col, x -> x > 10)           # Return a mask over the training set where :col > 10
    dataset |> training() |> mask(:col, x -> x > 10)

    dataset.filter(:col, x -> x > 10)             # Return a dataset where :col > 10
    dataset |> filter(:col, x -> x > 10)

    dataset.training.split(1, 1)                  # Further split a dataset with no splits
    dataset |> training() |> split(1, 1)
    ```
    """
    function SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true, seed=nothing)

        if seed === nothing
            rng = default_rng()
        elseif seed isa Int
            rng = MersenneTwister(seed)
        elseif seed isa AbstractRNG
            rng = seed
        else
            throw(ArgumentError("Seed must be nothing (default rng), an integer, or a random number generator"))
        end

        if subsample !== nothing
            if subsample isa Integer
                subsample >= nrow(df) && throw(ArgumentError("Subsample is greater than the number of rows in the DataFrame"))
                subsample <= 0 && throw(ArgumentError("Subsample must be greater than 0"))
            elseif subsample isa Float64 && !(0 < subsample <= 1)
                throw(ArgumentError("Subsample must be a fraction between 0 and 1"))
            else
                throw(ArgumentError("Subsample must be an integer or a fraction"))
            end
        end

        conflicts = intersect(propertynames(df), fieldnames(SMSDataset), _reserved)
        if !isempty(conflicts)
            @warn "Conflicting properties: $conflicts\nThose properties of the DataFrame will not be accessible."
        end

        if returncopy
            df = copy(df)
        end

        if shuffle
            df = shuffle!(rng, df)
        end

        if subsample isa Integer
            df = df[1:subsample, :]
        elseif subsample isa Float64
            df = df[1:round(Int, subsample*nrow(df)), :]
        end

        if splits === nothing || length(splits) <= 1
            @warn "Only 1 or no split specified, the dataset will not be split"
            __slices = nothing
        else
            bnds = cumsum(vcat(0, splits / sum(splits)))
            __slices = [round(Int, bnds[i]*nrow(df)+1):round(Int, bnds[i+1]*nrow(df)) for i in 1:length(splits)]
        end

        goodvalues = Base.filter(x -> (x isa Number) && isfinite(x))

        possiblynumeric = any.(map.(x -> (<:).(x, Number), gettypes.(eltype.(eachcol(df)))))
        twoormorefinite = length.(goodvalues.(eachcol(df))) .>= 2 # For standard deviation
        validmask = possiblynumeric .& twoormorefinite

        __zero = mapcols(col -> mean(Vector{Number}(goodvalues(col))), df[:, validmask])
        __scale = mapcols(col -> std(Vector{Number}(goodvalues(col))), df[:, validmask])

        return SMSDataset(df, __slices, __zero, __scale, rng)

    end

    function SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\t", shuffle=true, subsample=nothing, seed=nothing)
        return SMSDataset(DataFrame(CSV.File(csvfile, delim=delim)), splits=splits, shuffle=shuffle, subsample=subsample, returncopy=false, seed=seed)
    end

    function SMSDataset(dataset::SMSDataset; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true, seed=nothing)
        length(dataset) == 0 || throw(ArgumentError("Can only split a dataset with no splits"))
        __zero, __scale = dataset.__zero, dataset.__scale
        newdataset = SMSDataset(dataset.__df, splits=splits, shuffle=shuffle, subsample=subsample, returncopy=returncopy, seed=seed)

        # This is quite the corner case but
        # if the new dataset is empty, usually because of
        # an agressive filter() followed by a split,
        # the __zero and __scale fields will be empty.
        # Deal with both non-empty and empty cases in-place
        # since SMSDataset is immutable.
        append!(newdataset.__zero, __zero)
        append!(newdataset.__scale, __scale)
        if nrow(newdataset.__zero) > 1
            @assert nrow(newdataset.__scale) == 2
            delete!(newdataset.__zero, 1)
            delete!(newdataset.__scale, 1)
        end

        return newdataset
    end

    gettypes(u::Union) = [u.a; gettypes(u.b)]
    gettypes(u) = [u]

    function Base.show(io::IO, dataset::SMSDataset)
        print(io, "SMSDataset(")
        Base.show(io, dataset.__df)
        println()
        print(io, ") containing $(dataset.__slices !== nothing ? "$(length(dataset)) splits of sizes $(length.(dataset.__slices))" : "no splits")")
    end

    function Base.iterate(dataset::SMSDataset, state=1)
        (dataset.__slices === nothing || state > length(dataset.__slices)) && return nothing
        return dataset[state], state+1
    end

    function Base.getindex(dataset::SMSDataset, i::Int)
        isnothing(dataset.__slices) && throw(ArgumentError("Dataset has no splits"))
        return SMSDataset(dataset.__df[dataset.__slices[i], :], nothing, dataset.__zero, dataset.__scale)
    end

    Base.getindex(dataset::SMSDataset, ::Colon) = [s for s in dataset]

    function Base.getindex(dataset::SMSDataset, x...)
        return dataset.__df[x...]
    end

    function Base.length(dataset::SMSDataset)
        return dataset.__slices !== nothing ? length(dataset.__slices) : 0
    end

    function Base.firstindex(::SMSDataset)
        return 1
    end

    function Base.lastindex(dataset::SMSDataset)
        return length(dataset)
    end

    function Base.getproperty(dataset::SMSDataset, name::Symbol)
        if name in fieldnames(SMSDataset)
            return getfield(dataset, name)
        elseif name in _reserved
            return getfield(SplitMaskStandardize, name)(dataset)
        else
            return getproperty(dataset.__df, name)
        end
    end

    (dataset::SMSDataset)(cols::Symbol...) = length(cols) == 1 ? dataset.__df[:, cols[1]] : stack([dataset.__df[:, col] for col in cols], dims=1)
    val(cols::Symbol...) = (dataset::SMSDataset) -> dataset(cols...)

    function training(dataset::SMSDataset)
        dataset.__slices === nothing && throw(ArgumentError("Dataset has no splits"))
        return SMSDataset(dataset.__df[dataset.__slices[1], :], nothing, dataset.__zero, dataset.__scale, dataset.__rng)
    end

    function validation(dataset::SMSDataset)
        (dataset.__slices === nothing || (dataset.__slices !== nothing && length(dataset.__slices) < 3)) && throw(ArgumentError("Dataset has less than 3 splits"))
        return SMSDataset(dataset.__df[dataset.__slices[2], :], nothing, dataset.__zero, dataset.__scale, dataset.__rng)
    end

    function test(dataset::SMSDataset)
        (dataset.__slices === nothing || (dataset.__slices !== nothing && length(dataset.__slices) < 2)) && throw(ArgumentError("Dataset has less than 2 splits"))
        return SMSDataset(dataset.__df[dataset.__slices[length(dataset.__slices)], :], nothing, dataset.__zero, dataset.__scale, dataset.__rng)
    end

    training() = (dataset::SMSDataset) -> training(dataset)
    validation() = (dataset::SMSDataset) -> validation(dataset)
    test() = (dataset::SMSDataset) -> test(dataset)
    
    nth(dataset::SMSDataset, n::Int) = dataset[n]
    nth(n::Int) = (dataset::SMSDataset) -> nth(dataset, n)

    mask(dataset::SMSDataset, col::Symbol, by=Bool) = by.(dataset.__df[:, col])
    mask(dataset::SMSDataset) = (col::Symbol, by=Bool) -> mask(dataset, col, by)
    mask(col::Symbol, by=Bool) = (dataset::SMSDataset) -> mask(dataset, col, by)

    idx(dataset::SMSDataset, col::Symbol, by=Bool) = findall(mask(dataset, col, by))
    idx(dataset::SMSDataset) = (col::Symbol, by=Bool) -> idx(dataset, col, by)
    idx(col::Symbol, by=Bool) = (dataset::SMSDataset) -> idx(dataset, col, by)

    Base.filter(dataset::SMSDataset) = (col::Symbol, by) -> filter(dataset, col, by)
    Base.filter(col::Symbol, by) = (dataset::SMSDataset) -> filter(dataset, col, by)
    function Base.filter(dataset::SMSDataset, col::Symbol, by)
        _idx = idx(dataset, col, by)
        newdf = dataset.__df[_idx, :]
        if dataset.__slices === nothing
            return SMSDataset(newdf, nothing, dataset.__zero, dataset.__scale, dataset.__rng)
        else
            # Behold! The reslicing!
            sliceidx = intersect.(Ref(_idx), dataset.__slices)
            offsets = cumsum(vcat(0, length.(sliceidx)))[begin:end-1]
            unitranges = map(x->firstindex(x):lastindex(x), sliceidx)
            offset_unitranges = ((ur, s) -> (ur) .+ s).(unitranges, offsets)
            return SMSDataset(newdf, offset_unitranges, dataset.__zero, dataset.__scale, dataset.__rng)
        end
    end

    presence(dataset::SMSDataset, col::Symbol) = filter(dataset, col, Bool)
    presence(dataset::SMSDataset) = (col::Symbol) -> presence(dataset, col)
    presence(col::Symbol) = (dataset::SMSDataset) -> presence(dataset, col)

    presmask(dataset::SMSDataset, col::Symbol) = mask(dataset, col, Bool)
    presmask(dataset::SMSDataset) = (col::Symbol) -> presmask(dataset, col)
    presmask(col::Symbol) = (dataset::SMSDataset) -> presmask(dataset, col)

    presidx(dataset::SMSDataset, col::Symbol) = idx(dataset, col, Bool)
    presidx(dataset::SMSDataset) = (col::Symbol) -> presidx(dataset, col)
    presidx(col::Symbol) = (dataset::SMSDataset) -> presidx(dataset, col)

    absence(dataset::SMSDataset, col::Symbol) = filter(dataset, col, x -> !Bool(x))
    absence(dataset::SMSDataset) = (col::Symbol) -> absence(dataset, col)
    absence(col::Symbol) = (dataset::SMSDataset) -> absence(dataset, col)

    absmask(dataset::SMSDataset, col::Symbol) = mask(dataset, col, x -> !Bool(x))
    absmask(dataset::SMSDataset) = (col::Symbol) -> absmask(dataset, col)
    absmask(col::Symbol) = (dataset::SMSDataset) -> absmask(dataset, col)

    absidx(dataset::SMSDataset, col::Symbol) = idx(dataset, col, x -> !Bool(x))
    absidx(dataset::SMSDataset) = (col::Symbol) -> absidx(dataset, col)
    absidx(col::Symbol) = (dataset::SMSDataset) -> absidx(dataset, col)

    Base.split(dataset::SMSDataset, splits::Int...; shuffle=true, subsample=nothing) = SMSDataset(dataset, splits=collect(splits), shuffle=shuffle, subsample=subsample, returncopy=false, seed=dataset.__rng)
    Base.split(dataset::SMSDataset) = (splits::Int...; shuffle=true, subsample=nothing) -> split(dataset, splits...; shuffle=shuffle, subsample=subsample)
    Base.split(splits::Int...; shuffle=true, subsample=nothing) = (dataset::SMSDataset) -> split(dataset, splits...; shuffle=shuffle, subsample=subsample)

    standardize(dataset::SMSDataset) = (cols::Union{Symbol, Colon}...) -> standardize(dataset, cols...)
    standardize(cols::Union{Symbol, Colon}...) = (dataset::SMSDataset) -> standardize(dataset, cols...)
    standardize(dataset::SMSDataset, ::Colon) = standardize(dataset, propertynames(dataset.__zero)...)

    function standardize(dataset::SMSDataset, cols::Symbol...)
        newdf = copy(dataset.__df)
        newzero = copy(dataset.__zero)
        newscale = copy(dataset.__scale)
        for col in cols
            if col in propertynames(dataset.__scale)
                transform!(newdf, col => (x -> (x .- dataset.__zero[1, col]) ./ dataset.__scale[1, col]) => col)
                transform!(newzero, col => (x -> 0.0) => col)
                transform!(newscale, col => (x -> 1.0) => col)
            else
                @warn "Column $col is not numeric or doesn't contains 2 or more finite values and will not be standardized"
            end
        end

        return SMSDataset(newdf, dataset.__slices, newzero, newscale, dataset.__rng)
    end

end
