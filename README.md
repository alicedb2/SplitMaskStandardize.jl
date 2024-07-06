# SplitMaskStandardize.jl

Barebone wrapper around a DataFrame which simplifies boiler-plate operations
often used prior to feeding data to ecological SDM models.

It can quickly generate training, validation and test splits,
extract presence and absence data, standardize the data, and filter
or return masks and index arrays according to given conditions.

The syntax is inspired by object oriented programming but is not.

## Basic usage
```julia
dataset = SMSDataset("data.csv")
# Defaults to splitting 1/3 training, 1/3 validation, and 1/3 test set.
# Splits can be specified by any vector of number which will be normalized.
dataset = SMSDataset("data.csv", splits=[2, 1, 1])
# The dataset is now split 50%, 25%, 25%.

training = dataset.training
validation = dataset.validation
test = dataset.test

julia> dataset(:col)
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
───── ──┼──────────────────────────────────────────────────────────────────
      1 │  -84.25    33.4167      1.0      1.0      1.0      1.0      1.0 ⋯
      2 │  -95.25    35.75        1.0      1.0      1.0      0.0      1.0
    ⋮   │     ⋮         ⋮        ⋮        ⋮        ⋮        ⋮        ⋮    ⋱
  39024 │ -108.25    28.0833      0.0      0.0      0.0      0.0      0.0
)
```
## Examples of chaining
```julia
dataset.presence(:species)
dataset.absence(:species)
# Return a dataset including only presence/absences of :species
# A Bool/!Bool is cast over the column :species so it
# must contain true/false/1/0

dataset.absence(:species)(:col1, :col2, :col3, :col4)
# Return a 4xN matrix of predictors :col1, :col2, :col3, :col4
# associated with absences of :species

dataset.standardize(:col1, :col2, :col3)
# Return a copy of the dataset where :col1, :col2, and :col3
# have been standardized against the training set

dataset.standarize(:col1, :col2)(:col1, :col2, :col3)
# Return a 3xN matrix of stacked columns across all splits
# where col1 and col2 have been standardized against the training set

dataset.validation.presence(:species).standardize(:col1, :col2)
dataset.validation.absence(:species).standardize(:col1, :col2)
# Return a dataset containing a copy of the underlying dataframe 
# of standardized predictors :col1 and :col2 for the presences 
# and absences of :species in the validation set 

# Naturally the following evaluates to true
 all(dataset.standardize(:BIO1, :BIO2).training(:BIO1, :BIO2, :BIO3)
 .== dataset.training.standardize(:BIO1, :BIO2)(:BIO1, :BIO2, :BIO3))

dataset.training.presence(:species1).presence(:species2)
# Return a dataset with rows where both :species1 and :species2 are present simultaneously

dataset.test.presence(:species1).absence(:species1)
# Return empty dataset since species1 can't be both absence and present

dataset.idx(:col, ismissing)             # Return indices of missing values in :col
dataset.test.presidx(:species)           # Return indices of presences of :species in the test set
dataset.training.absidx(:species)        # Return indices of absences of :species in the training set

dataset.presmask(:species)               # Return a mask of presences of :species in the dataset
dataset.absmask(:species)                # Return a mask of absences of :species in the dataset

dataset.training.mask(:col, x -> x > 10) # Return a mask of training set where :col > 10
dataset.filter(:col, x -> x > 10)        # Return a dataset where :col > 10

dataset.training.split(1, 1)             # Further split a dataset with no splits
dataset[6].split(2, 1, 1)                # The standardization is preserved across 
SMSDataset(dataset[4], splits=[1, 1])    # the splits by default.
                                         # Passing conservestandardization=false 
                                         # will cause the data to be restandardized
                                         # against the first split.
```