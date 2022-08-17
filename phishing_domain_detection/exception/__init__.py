import os
import sys

class Phishing_Exception(Exception):
    
    def __init__(self, error_message: Exception, error_details: sys) -> None:
        """ Initializer for our custom exception class

        Args:
            error_message (Exception): The message generated because of some exception. This message will be logged.
            error_details (sys): System related information. Helps in locating which file is causing error in which line.
        """
        
        super().__init__(error_message) # calling parent class constructor and passing error message
        self.error_message = Phishing_Exception.get_detailed_error_message(error_message, error_details)
    
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """generates the custom message for our exception

        Args:
            error_message (Exception): The exception object
            error_details (sys): The sys module object

        Returns:
            str: The generated custom exception message
        """
        _,_, exec_tb = error_details.exc_info() # returns type, value, traceback, Return information about the most recent exception caught by an except clause in the current stack frame or in an older stack frame.
        
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        
        file_name = exec_tb.tb_frame.f_code.co_filename
        
        error_message = f"Error occured in script: [ {file_name} ] at try block line number: [ {try_block_line_number} ] at exception block line number: [ {exception_block_line_number} ] error message: [{error_message}]"
        
        return error_message
    
    def __str__(self):
        return self.error_message
        
    def __repr__(self) -> str:
        return Phishing_Exception.__name__.str()