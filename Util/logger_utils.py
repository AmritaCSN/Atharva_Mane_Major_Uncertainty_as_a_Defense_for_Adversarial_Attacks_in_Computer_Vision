import logging

def get_logger(filename):
    # Create a logger if it doesn't exist
    logger = logging.getLogger(filename)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create a file handler
        file_handler = logging.FileHandler(filename, mode='w')  # Set mode to 'w' to clear the file
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger
        logger.addHandler(file_handler)

        # Add a StreamHandler to log to console as well
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
