module SplitMaskStandardize
    
    using DataFrames
    using CSV
    using StatsBase: std, mean
    using Random: shuffle!
    
    import Base: filter

    export SMSDataset

    struct SMSDataset
        __df::AbstractDataFrame
        __slices::Union{Nothing, Vector{UnitRange{Int64}}}
        __zero::AbstractDataFrame
        __scale::AbstractDataFrame
    end

    """
    SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
    SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\\t", shuffle=true, subsample=nothing)
    
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

    Splits can be specified either as a vector of any numbers
    which are normalized to fractions.
    For example [2, 1, 2] will result in [0.4, 0.2, 0.4] splits.

    When 3 splits are specified, the dataset is split into
    training, validation, and test sets.

    When 2 splits are specified the dataset is split into
    training and test sets.

    When an arbitrary number of splits are specified, the first 
    and last split is considered as the training and test sets.
    The splits can be access using an index, i.e. dataset[i]
    
    Properties/fields of the underlying DataFrames are exposed that are not
    "presence", "absence", "standardize", "training", "validation", "test", "filter",
    or internal fields "__df", "__slices", "__zero", "__scale".

    # Basic examples
    ```julia-repl
    dataset = SMSDataset("data.csv")
    training = dataset.training
    validation = dataset.validation
    test = dataset.test
    nth_split = dataset[n]

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

    ```
    """
    function SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
        
        sum(splits) > 0 || throw(ArgumentError("At least one split must be greater than 0"))

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

        conflicts = intersect(propertynames(df), [:training, :validation, :test, :filter, :presence, :absence, :standardize, :df, :__slices, :__zero, :__scale])
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
        
        bnds = cumsum(vcat(0, splits / sum(splits)))
        __slices = [round(Int, bnds[i]*nrow(df)+1):round(Int, bnds[i+1]*nrow(df)) for i in 1:length(splits)]

        __zero = mapcols(mean, df)
        __scale = mapcols(std, df)

        return SMSDataset(df, __slices, __zero, __scale)

    end

    function Base.show(io::IO, dataset::SMSDataset)
        print(io, "SMSDataset(")
        Base.show(io, dataset.__df)
        print(io, ")")
    end

    function SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\t", shuffle=true, subsample=nothing) 
        return SMSDataset(DataFrame(CSV.File(csvfile, delim=delim)), splits=splits, shuffle=shuffle, subsample=subsample, returncopy=false)
    end

    function Base.getindex(dataset::SMSDataset, i::Int)
        return dataset.__df[dataset.__slices[i], :]
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
        elseif name === :filter
            return filter(dataset)
        elseif name === :presence
            return presence(dataset)
        elseif name === :absence
            return absence(dataset)
        elseif name === :standardize
            return standardize(dataset)
        else
            return getproperty(dataset.__df, name)
        end
    end

    (dataset::SMSDataset)(cols::Symbol...) = length(cols) == 1 ? dataset.__df[:, cols[1]] : stack([dataset.__df[:, col] for col in cols], dims=1)
    
    filter(dataset::SMSDataset) = (col::Symbol, by=Bool) -> SMSDataset(dataset.__df[by.(dataset.__df[:, col]), :], dataset.__slices, dataset.__zero, dataset.__scale)

    presence(dataset::SMSDataset) = (col::Symbol) -> filter(dataset)(col, Bool)

    absence(dataset::SMSDataset) = (col::Symbol) -> filter(dataset)(col, x -> !Bool(x))

    function standardize(dataset::SMSDataset)
        function fun(cols::Symbol...)
            length(cols) > 0 || throw(ArgumentError("At least one column must be provided"))
            ret = copy(dataset.__df)
            for col in cols
                transform!(ret, col => (x -> (x .- dataset.__zero[1, col]) ./ dataset.__scale[1, col]) => col)
            end
            return SMSDataset(ret, dataset.__slices, dataset.__zero, dataset.__scale)
        end
        return fun
    end

end
