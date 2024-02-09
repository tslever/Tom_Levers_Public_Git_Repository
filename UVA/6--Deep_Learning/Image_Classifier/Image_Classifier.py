import tensorflow

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Physical Devices:\n{physical_devices}")
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print(f"There are no physical devices.")
