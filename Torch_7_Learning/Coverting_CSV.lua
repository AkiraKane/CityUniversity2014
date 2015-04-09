-- Importing the Bank Marketing Dataset
-- Daniel Dixey
-- Started: 9 April 2015

-- Loading CSV Files

print('============================================================')
print('Constructing dataset')
print('')

require 'csvigo'

local function import_and_convert()
    -- Reading CSV files can be tricky. This code uses the csvigo package for this:
    Training = csvigo.load{path = 'Training.csv', mode = 'tidy'}

    --print('')
    --print("Feature size: ", #Training)
    --print("Number of Samples: ", #Training[1])
    --print('')

    col = 1
    ind = 1
    for val, values in pairs(Training) do
        if col == 64 then
            dataset_outputs = torch.Tensor(values)
        else
            if col == 1 then
                temp = torch.Tensor(values)
                dataset_inputs = torch.Tensor((#temp)[1], 63 )
                print(dataset_inputs:nDimension())
                dataset_inputs[{ {},ind }] = temp
                ind = ind + 1
        else
                temp = torch.Tensor(values)
                dataset_inputs[{ {},ind }] = temp
                ind = ind + 1
            end
        end
        col = col + 1
    end

    -- Save to Torch File
    torch.save('Training_Data.th7', dataset_inputs)
    torch.save('Training_Labels.th7', dataset_outputs)

    return 'Complete'
end

local function main()
    Import = import_and_convert()
    print(Import)
end

main()
