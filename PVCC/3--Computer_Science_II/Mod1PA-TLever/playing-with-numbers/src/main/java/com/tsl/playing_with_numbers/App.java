package com.tsl.playing_with_numbers;


import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;


/**
 * App encapsulates the entry point to a program to fill and display a faculty member's schedule of office hours.
 * @author Tom Lever
 * @version 1.0
 * @since 05/22/21
 *
 */
class App 
{
	
	/**
	 * main is the entry point of the program to fill and display a faculty member's schedule of office hours.
	 * @param args
	 */
    public static void main( String[] args )
    {
    	int theNumberOfOfficeHours = 6;
    	
        String[] theStudentsNames = new String[theNumberOfOfficeHours];
        
        HashMap<Integer, Integer> theOfficeHoursAndIndicesInTheArrayOfStudentsNames =
            getTheOfficeHoursAndIndicesInTheArrayOfStudentsNames(theNumberOfOfficeHours);
        
        AnInputOutputManager theInputOutputManager = new AnInputOutputManager();
        
        int theNumberOfStudentsWhoHaveSignedUp = 0;
        
        while (theNumberOfStudentsWhoHaveSignedUp < theNumberOfOfficeHours) {
        	
        	try {
        		
        		AnIntegerAndAString theProposedTimeAndName = theInputOutputManager.askAboutAndReadATimeAndAName();
        		
        		int theProposedTime = theProposedTimeAndName.getTheInteger();
        		
        		if (theStudentsNames[theOfficeHoursAndIndicesInTheArrayOfStudentsNames.get(theProposedTime)] != null) {
        			throw new TimeInUseException("The time " + theProposedTime + " PM is already in use.");
        		}
        		
        		theStudentsNames[theOfficeHoursAndIndicesInTheArrayOfStudentsNames.get(theProposedTime)] =
        			theProposedTimeAndName.getTheString();
        		
            	theNumberOfStudentsWhoHaveSignedUp++;
        		
        	}
        	
        	catch (InputMismatchException theInputMismatchException) {
        		System.out.println("Input mismatch exception.");
        	}
        	
        	catch (NoSuchElementException theNoSuchElementException) {
        		System.out.println("No such element exception.");
        	}
        	
        	catch (AnInvalidTimeException theInvalidTimeException) {
        		System.out.println(theInvalidTimeException.getMessage());
        	}
        	
        	catch (AnInvalidNameException theInvalidNameException) {
        		System.out.println(theInvalidNameException.getMessage());
        	}
        	
        	catch (TimeInUseException theTimeInUseException) {
        		System.out.println(theTimeInUseException.getMessage());
        	}
        	
        }
        
        displayTheScheduleOfOfficeHours(
        	theNumberOfOfficeHours, theOfficeHoursAndIndicesInTheArrayOfStudentsNames, theStudentsNames);
        
    }
    
    
    /**
     * getTheOfficeHoursAndIndicesInTheArrayOfStudentNames provides a map of integer office hours and indices in the
     * array of student names.
     * @param theNumberOfOfficeHours
     * @return
     */
    private static HashMap<Integer, Integer> getTheOfficeHoursAndIndicesInTheArrayOfStudentsNames(
    	int theNumberOfOfficeHours
    ) {
    	
        HashMap<Integer, Integer> theOfficeHoursAndIndicesInTheArrayOfStudentsNames = new HashMap<Integer, Integer>();
        
        for (int i = 0; i < theNumberOfOfficeHours; i++) {
        	theOfficeHoursAndIndicesInTheArrayOfStudentsNames.put(i + 1, i);
        }
        
        return theOfficeHoursAndIndicesInTheArrayOfStudentsNames;
    	
    }
    
    
    private static void displayTheScheduleOfOfficeHours(
    	int theNumberOfOfficeHours,
    	HashMap<Integer, Integer> theOfficeHoursAndIndicesInTheArrayOfStudentsNames,
    	String[] theStudentsNames
    ) {
    	
        System.out.println(
        	"Schedule is full. No appointments available!\n" +
        	"Schedule is:"
        );
        
        for (int i = 0; i < theNumberOfOfficeHours; i++) {
        	System.out.println(
        		(theOfficeHoursAndIndicesInTheArrayOfStudentsNames.keySet().toArray())[i] +
        		" PM: " +
        		theStudentsNames[i]
        	);
        }
    	
    }
    
}
