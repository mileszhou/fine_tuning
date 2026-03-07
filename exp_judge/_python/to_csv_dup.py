import json
from pathlib import Path
from io import StringIO

from numpy import mean, sqrt, std
from sympy import false

# ----------- CONFIG ------------- 
# Parameters for the experiment, used for constructing input and output file names, and for debugging
T= 0.2
Replication= 10

# Files involved in the process, with paths constructed from the parameters above
input_dir = "./_results/judge"
input_fn = f"judgement_T={T}_dup({Replication}).jsonl"     # input file with judgement and scores from the training comparison step
input_path = Path(input_dir) / input_fn
results_dir = "./_results/judge"
result_fn = f"judgement_T={T}_average.jsonl"
result_path = Path(results_dir) / result_fn
# --------------------------------

# Define constants for attribute names
WINNER = "winner"           # The winner of the comparison, 1 for model 1, 2 for model 2, never ties
WINNING = "winning"
CONFIDENCE= "confidence"
ACCURACY = "accuracy"
SAFETY = "safety"
COMPLETENESS = "completeness"
SCORES = "scores"

OVERALL = (         # Overall attributes, returned by the model
    WINNING,
    CONFIDENCE
)

DATA_COLUMNS = (    # Performce attributes, returned by the evaluation function
    ACCURACY,
    SAFETY,
    COMPLETENESS
)

DELTA_COLUMNS_DICT = {f"{key}": f"{key}_delta" for key in DATA_COLUMNS}
DELTA_COLUMNS = tuple(DELTA_COLUMNS_DICT.values())

STDDEV_COLUMNS_DICT = {f"{key}": f"{key}_stddev" for key in DATA_COLUMNS}
STDDEV_COLUMNS = tuple(STDDEV_COLUMNS_DICT.values())

ALL_COLUMNS = OVERALL + DATA_COLUMNS + DELTA_COLUMNS + STDDEV_COLUMNS

IDX = "idx"     # Index of the record, used for grouping records into batches (records with the same index belong to the same batch)
NN = "nn"       # n/NN is the number of records in the batch, used for calculating averages and variances

class Batch:
    __slots__ = ("data", "closed", "square_sum")

    def __init__(self, idx=-1):
        self.closed = False
        self.data = {
            IDX: -1,    # Index of the batch, used for grouping records into batches (records with the same index belong to the same batch)
            NN: 0,       # n/NN is the number of records in the batch, used for calculating averages and variances
            **{key: 0.0 for key in ALL_COLUMNS}   # Initialize all overall, data, delta and stddev columns to 0.0
        }
        self.square_sum = {key: 0.0 for key in DATA_COLUMNS}
    
    def add_item(self, item):
        if self.closed:
            raise ValueError("Attempting to add item to a closed batch")
        if self.data[IDX]>=0 and self.data[IDX] != item[IDX]:
            raise ValueError(f"Attempting to add item with index {item[IDX]} to batch with index {self.data[IDX]}")
        
        # Actually adding an item
        data = self.data
        square_sum = self.square_sum

        data[IDX] = item[IDX]
        data[NN] += 1

        data[WINNING] += (item[WINNER]-1.5)*2*item[CONFIDENCE]  # [1-2] to [-1,1] --> winning
        data[CONFIDENCE] += item[CONFIDENCE]

        for key in DATA_COLUMNS:
            data[key] += item[SCORES][0][key]
            delta = item[SCORES][1][key] - item[SCORES][0][key]   # Difference
            data[DELTA_COLUMNS_DICT[key]] += delta
            square_sum[key] += delta*delta

    def close(self):
        n = self.data[NN]
        data = self.data
        square_sum = self.square_sum
            
        if n > 0:
            inv_n = 1.0 / n
            if data[CONFIDENCE] == 0.0:
                data[WINNING] = 0.0
            else:
                data[WINNING] /= data[CONFIDENCE]   # Re-calculate winning score as average winning score per confidence point
                data[CONFIDENCE] *= inv_n                  # Now it's average confidence, which is useless, but we can use it for debugging
            
            for key in DATA_COLUMNS + DELTA_COLUMNS:
                data[key] *= inv_n

            if n>1:
                for key in DATA_COLUMNS:
                    x_bar = data[DELTA_COLUMNS_DICT[key]]   # The average difference, which is the mean of the distribution we want to calculate std_dev for
                    sq_sum = square_sum[key]
                    data[STDDEV_COLUMNS_DICT[key]] = sqrt((sq_sum - n * x_bar * x_bar) / (n-1))
            else: # n==1
                for key in DATA_COLUMNS:
                    data[STDDEV_COLUMNS_DICT[key]] = 0.0
                    data[DELTA_COLUMNS_DICT[key]] = 0.0
        
        self.closed = True
        return data

class Column_Stats:
    __slots__ = ("data", "square_sum", "closed")

    def __init__(self):
        self.closed = False         # Control overall status and the interpretation of sums/average
        self.data = {
            NN: 0,
            **{key: 0.0 for key in ALL_COLUMNS}   # Sums of each column, used for calculating averages and variances
        }
        self.square_sum = {key: 0.0 for key in DATA_COLUMNS}
    
    def add_batch(self, batch):
        if not isinstance(batch, Batch):
            raise ValueError("Attempting to add a non-Batch object")
        if not batch.closed:
            raise ValueError("Attempting to add an open batch")

        # Okay, we have a closed batch, we can add its data to the stats
        batch_n = batch.data[NN]
        batch_data = batch.data
        batch_square = batch.square_sum
        
        data = self.data
        squares = self.square_sum
        data[NN] += batch_n

        data[WINNING] += batch_data[WINNING] * batch_data[CONFIDENCE] * batch_n     # Recover the sum from the average
        data[CONFIDENCE] += batch_data[CONFIDENCE]*batch_n     # Recover the sum from the average

        for key in DATA_COLUMNS + DELTA_COLUMNS:
            data[key] += batch_data[key]*batch_n     # Recover the sum from the average
        for key in DATA_COLUMNS:
            squares[key] += batch_square[key]

    def close(self):
        if self.closed:
            raise ValueError("Attempting to close an already closed Column_Stats")

        data = self.data
        n = data[NN]        

        if n > 0:
            inv_n = 1.0 / n
            std_dev = {key: 0.0 for key in DATA_COLUMNS}
            data = self.data
            squares = self.square_sum

            if data[CONFIDENCE] == 0.0:
                data[WINNING] = 0.0              # 0, if no confidence at all
            else:
                data[WINNING] /= data[CONFIDENCE]
                data[CONFIDENCE] *= inv_n       # Now, it's average confidence, which is useless, but we can use it for debugging

            if n > 1:
                for key in DATA_COLUMNS:
                    data[key] *= inv_n
                    data[DELTA_COLUMNS_DICT[key]] *= inv_n

                    x_bar = data[DELTA_COLUMNS_DICT[key]]
                    data[STDDEV_COLUMNS_DICT[key]] = sqrt((squares[key] - n * x_bar * x_bar) / (n-1))

        closed = True
        return data

def read_obj(fi):
    line = fi.readline()
    return None if line is None or line=="" or line.strip()=="" else json.loads(line)

fi = open(input_path, "r", encoding="utf-8")
fo = open(result_path, "w", encoding="utf-8")

def main():
    n_batches = 0
    current_idx = -1        # All valid index should be non-negative
    current_batch = Batch() # Start with empty batch, we will add items to it until we encounter a new batch or EoF
    column_stats = Column_Stats()   # Start with empty column stats

    done = False
    while not done:
        item = read_obj(fi)
        done = item is None
        if (item is None) or (current_idx>=0 and current_idx!=item[IDX]):   # EoF or new batch starts: close current batch and add it to column stats
            n_batches += 1
            data = current_batch.close()
            json.dump(data, fo)
            fo.write("\n")
            column_stats.add_batch(current_batch)
            current_batch = Batch()
            json.dump(column_stats.data, fo)   # Debugging: print current column stats after each batch is closed
            fo.write("\n")
            # Continue to add the new item to the new batch if it's not EoF, otherwise we are done
        if item:    # Not EoF, we can add it to the current batch
            current_idx = item[IDX]
            current_batch.add_item(item)
    json.dump(column_stats.close(), fo)
    fo.write("\n")
    
    fi.close()
    fo.close()
    print(f"Done! {n_batches} records saved to {result_path}.\n")

main()
