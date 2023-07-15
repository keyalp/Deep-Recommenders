from preprocessing import BasicPreprocessing, NetflixData

basic_pre = BasicPreprocessing()

# Execute the basic preprocessing:
data = basic_pre.step_1(basic_pre.data)
data = basic_pre.step_2(data)
data = basic_pre.step_3(data)
train, test = basic_pre.step_4(data)

# Write the datasets because we will use this ones for the factorixation machines and for the absolute popularity model
train.to_csv("data/Subset1M_traindata.csv", index=False, header=False)
test.to_csv("data/Subset1M_testdata.csv", index=False, header=False)

netflix_data = NetflixData(train, test)

netflix_data.create_negative_samples()
netflix_data.split()

netflix_data.train.to_csv("data/train.csv")
netflix_data.validation.to_csv("data/validation.csv")
netflix_data.test.to_csv("data/test.csv")

print("Train, validation and test datasets have been successfully stored in the data folder")


