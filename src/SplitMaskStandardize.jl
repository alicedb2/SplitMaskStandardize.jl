module SplitMaskStandardize

    using DataFrames
    using CSV
    using StatsBase: std, mean
    using Random: shuffle!

    export SMSDataset

    struct SMSDataset
        __df::AbstractDataFrame
        __slices::Union{Nothing, Vector{UnitRange{Int64}}, Vector{Vector{Int64}}}
        __zero::AbstractDataFrame
        __scale::AbstractDataFrame
    end

    """
    SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
    SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\\t", shuffle=true, subsample=nothing)
    SMSDataset(dataset::SMSDataset; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true, conservestrandardization=true)
    
    Create a dataset object from a DataFrame or a CSV file.

    The dataset object is a barebone wrapper over a DataFrame which
    allows to quickly generate training, validation and test splits,
    extract presence and absence data, and standardize the data.
    The syntax is inspired by object oriented programming but is not.

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
    training = dataset.training
    validation = dataset.validation
    test = dataset.test
    nth_split = dataset[n]
    splits = [split for split in dataset]

    julia> dataset(:col)             # Return a column from the underlying DataFrame as a vector
    39024-element Vector{Float64}:
     25.518239974975586
     10.868968963623047
      ⋮
      2.759552240371704

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

    julia> dataset.filter(:sp1, x -> x > 10)
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
    dataset.absence(:column)   # :column must contain true/false/1/0 values
                               # false/0 are considered as an absence

    dataset.absence(:species)(:col1, :col2, :col3, :col4)   # return a 4xN matrix of predictors associated
                                                            # with absences of :species

    dataset.standardize(:col1, :col2, :col3)   # Return a copy of the dataset where col1, col2, and col3
                                               # have been standardized against the training set

    dataset.standarize(:col1, :col2)(:col1, :col2, :col3)   # Return a 3xN matrix of stacked columns across all splits
                                                            # where col1 and col2 have been standardized against the training set

    dataset.validation.presence(:species).standardize(:col1, :col2)   # Return dataset containing a copy of the underlying
    dataset.validation.absence(:species).standardize(:col1, :col2)    # dataframe at presences or absences of :species in the
                                                                      # validation set where col1 and col2 have been
                                                                      # standardized against the training set

    dataset.training.presence(:species1).presence(:species2)   # Return simultaneous presences of both species in training set

    dataset.test.presence(:species1).absence(:species1)        # Return empty dataset

    dataset.idx(:col, ismissing)                  # Return indices of missing values in :col
    dataset.test.presidx(:species)                # Return indices of presences of :species in the test set
    dataset.training.absidx(:species)             # Return indices of absences of :species in the training set

    dataset.presmask(:species)                    # Return a mask of presences of :species in the dataset
    dataset.absmask(:species)                     # Return a mask of absences of :species in the dataset

    dataset.training.mask(:col, x -> x > 10)      # Return a mask of training set where :col > 10
    dataset.filter(:col, x -> x > 10)             # Return a dataset where :col > 10

    dataset.training.split(1, 1)                 # Further split a dataset with no splits
    SMSDataset(dataset[4], splits=[1, 1])        # The standardization is preserved across the splits by default
                                                 # but could be turned off by passing conservestandardization=false
    ```
    """
    function SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)

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

        _reserved = [:training, :validation, :test,
                     :idx, :mask, :filter, :split,
                     :presence, :absence, :standardize,
                     :presidx, :absidx, :presmask, :absmask,
                     :__df, :__slices, :__zero, :__scale]

        conflicts = intersect(propertynames(df), _reserved)
        if !isempty(conflicts)
            @warn "Conflicting properties: $conflicts\nThose properties of the DataFrame will not be accessible."
        end

        if returncopy
            df = copy(df)
        end

        if shuffle
            df = shuffle!(df)
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

        goodvalues = Base.filter(x -> x !== nothing && x isa Number && isfinite(x))

        possiblynumeric = any.(map.(x -> (<:).(x, Number), gettypes.(eltype.(eachcol(df)))))
        twoormorefinite = length.(goodvalues.(eachcol(df))) .>= 2 # For standard deviation
        validmask = possiblynumeric .& twoormorefinite

        __zero = mapcols(col -> mean(Vector{Number}(goodvalues(col))), df[:, validmask])
        __scale = mapcols(col -> std(Vector{Number}(goodvalues(col))), df[:, validmask])

        return SMSDataset(df, __slices, __zero, __scale)

    end

    function SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\t", shuffle=true, subsample=nothing)
        return SMSDataset(DataFrame(CSV.File(csvfile, delim=delim)), splits=splits, shuffle=shuffle, subsample=subsample, returncopy=false)
    end

    function SMSDataset(dataset::SMSDataset; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true, conservestrandardization=true)
        length(dataset) == 0 || throw(ArgumentError("Can only split a dataset with no splits"))
        __zero, __scale = dataset.__zero, dataset.__scale
        newdataset = SMSDataset(dataset.__df, splits=splits, shuffle=shuffle, subsample=subsample, returncopy=returncopy)
        if conservestrandardization
            newdataset.__zero .= __zero
            newdataset.__scale .= __scale
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
        if name in [:__df, :__slices, :__zero, :__scale]
            return getfield(dataset, name)
        elseif name === :training
            return SMSDataset(dataset.__df[dataset.__slices[1], :], nothing, dataset.__zero, dataset.__scale)
        elseif name === :validation
            length(dataset.__slices) < 3 && throw(ArgumentError("Dataset has less than 3 splits"))
            return SMSDataset(dataset.__df[dataset.__slices[2], :], nothing, dataset.__zero, dataset.__scale)
        elseif name === :test
            return SMSDataset(dataset.__df[dataset.__slices[length(dataset.__slices)], :], nothing, dataset.__zero, dataset.__scale)
        elseif name === :idx
            return idx(dataset)
        elseif name === :mask
            return mask(dataset)
        elseif name === :filter
            return filter(dataset)
        elseif name === :presence
            return presence(dataset)
        elseif name === :absence
            return absence(dataset)
        elseif name === :presmask
            return presmask(dataset)
        elseif name === :absmask
            return absmask(dataset)
        elseif name === :standardize
            return standardize(dataset)
        elseif name === :split
             return split(dataset)
        else
            return getproperty(dataset.__df, name)
        end
    end

    (dataset::SMSDataset)(cols::Symbol...) = length(cols) == 1 ? dataset.__df[:, cols[1]] : stack([dataset.__df[:, col] for col in cols], dims=1)

    idx(dataset::SMSDataset) = (col::Symbol, by=Bool) -> findall(by.(dataset.__df[:, col]))
    mask(dataset::SMSDataset) = (col::Symbol, by=Bool) -> by.(dataset.__df[:, col])
    
    function Base.filter(dataset::SMSDataset)
        function fun(col::Symbol, by=Bool)
            _idx = idx(dataset)(col, by)
            newdf = dataset.__df[_idx, :]
            if dataset.__slices === nothing
                return SMSDataset(newdf, nothing, dataset.__zero, dataset.__scale)
            else
                # Behold! The reslicing!
                sliceidx = intersect.(Ref(_idx), dataset.__slices)
                offsets = cumsum(vcat(0, length.(sliceidx)))[begin:end-1]
                unitranges = map(x->firstindex(x):lastindex(x), sliceidx)
                offset_unitranges = ((ur, s) -> (ur) .+ s).(unitranges, offsets)
                return SMSDataset(newdf, offset_unitranges, dataset.__zero, dataset.__scale)
            end
        end
        return fun
    end

    presence(dataset::SMSDataset) = (col::Symbol) -> filter(dataset)(col, Bool)
    presmask(dataset::SMSDataset) = (col::Symbol) -> mask(dataset)(col, Bool)
    presidx(dataset::SMSDataset) = (col::Symbol) -> idx(dataset)(col, Bool)

    absence(dataset::SMSDataset) = (col::Symbol) -> filter(dataset)(col, x -> !Bool(x))
    absmask(dataset::SMSDataset) = (col::Symbol) -> mask(dataset)(col, x -> !Bool(x))
    absidx(dataset::SMSDataset) = (col::Symbol) -> idx(dataset)(col, x -> !Bool(x))

    split(dataset::SMSDataset) = (splits...; shuffle=true, subsample=nothing, conservestandardization=true) -> SMSDataset(dataset, splits=collect(splits), shuffle=shuffle, subsample=subsample, conservestandardization=conservestandardization)

    function standardize(dataset::SMSDataset)
        function fun(cols::Symbol...)
            length(cols) > 0 || throw(ArgumentError("At least one column must be provided"))
            ret = copy(dataset.__df)
            for col in cols
                if col in propertynames(dataset.__scale)
                    transform!(ret, col => (x -> (x .- dataset.__zero[1, col]) ./ dataset.__scale[1, col]) => col)
                else
                    @warn "Column $col is not numeric or doesn't contains 2 or more finite values and will not be standardized"
                end
            end
            return SMSDataset(ret, dataset.__slices, dataset.__zero, dataset.__scale)
        end
        return fun
    end

end
