# SplitMaskStandardize.jl

```
  SMSDataset(df::AbstractDataFrame; splits=[1/3, 1/3, 1/3], shuffle=true, subsample=nothing, returncopy=true)
  SMSDataset(csvfile::AbstractString; splits=[1/3, 1/3, 1/3], delim="\t", shuffle=true, subsample=nothing)

  Create a dataset object from a DataFrame or a CSV file.

  The dataset object is a barebone wrapper over a DataFrame which allows to quickly generate training, validation and test splits, extract
  presence and absence data, and standardize the data. The syntax is inspired by object oriented programming but is not.

  Arguments:
  ≡≡≡≡≡≡≡≡≡≡

    •  df: DataFrame object.

    •  csvfile: Path to a CSV file.

    •  splits: Fractions (will be normalized) of the dataset to split into training, validation, and test sets

    •  delim: Delimiter for the CSV file.

    •  shuffle: Shuffle the dataset before splitting.

    •  subsample: Number (integer) or fraction (float) of rows to subsample from the dataset.

    •  returncopy: Return a copy of the DataFrame object.

  Splits can be specified either as a vector of any numbers which are normalized to fractions. 
  For example [2, 1, 2] will result in [0.4, 0.2, 0.4] splits.

  When 3 splits are specified, the dataset is split into training, validation, and test sets.

  When 2 splits are specified the dataset is split into training and test sets.

  When an arbitrary number of splits are specified, the first and last split is considered as 
  the training and test sets. The splits can be access using an index, i.e. dataset[i]

  Properties/fields of the underlying DataFrames are exposed that are not 
  "presence", "absence", "standardize", "training", "validation", "test", 
  "__df", "__slices", "__zero", "__scale".

  Examples:
  ≡≡≡≡≡≡≡≡≡

  dataset = SMSDataset("data.csv")
  training = dataset.training
  validation = dataset.validation
  test = dataset.test
  
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
  )
  
  
  # By default presence(:col) and absence(:col) mask by casting Bool on 
  # the elements of :col, but a custom mask can be provided in case the 
  # column contains some other type of values.

  julia> dataset.presence(:sp1, x -> x > 10)
  SMSDataset(39024×81 DataFrame
     Row │ lon        lat      sp1      sp2      sp3      sp4      sp5     ⋯
         │ Float64    Float64  Float64  Float64  Float64  Float64  Float64 ⋯
  ───────┼──────────────────────────────────────────────────────────────────
       1 │  -84.25    33.4167      1.0      1.0      1.0      1.0      1.0 ⋯
       2 │  -95.25    35.75        1.0      1.0      1.0      0.0      1.0
     ⋮   │     ⋮         ⋮        ⋮        ⋮        ⋮        ⋮        ⋮    ⋱
   39024 │ -108.25    28.0833      0.0      0.0      0.0      0.0      0.0
  )
  ```  

  Examples of chaining
  ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

  dataset.absence(:column)   # :column must contain true/false/1/0 values
                             # false/0 are considered as an absence
  
  dataset.absence(:species)(:col1, :col2, :col3, :col4)   # return a 4xN matrix of predictors associated
                                                          # with absences of :species
  
  dataset.standardize(:col1, :col2, :col3)   # Return a 3xN matrix of predictors standardized
                                             # against the training set mean and standard deviation
  
  dataset.validation.presence(:species).standardize(:col1, :col2)   # Return 2xN matrix of predictors associated
  dataset.validation.absence(:species).standardize(:col1, :col2)    # with presences or absences of :species in the
                                                                    # validation set standardized against the training set
  # Note that standardize() does not return an SMSDataset
  # and therefore prevents further chaining
  
  dataset.training.presence(:species1).presence(:species2)   # Return simultaneous presences of both species in training set
  
  dataset.test.presence(:species1).absence(:species1)        # Return empty dataset
  ```