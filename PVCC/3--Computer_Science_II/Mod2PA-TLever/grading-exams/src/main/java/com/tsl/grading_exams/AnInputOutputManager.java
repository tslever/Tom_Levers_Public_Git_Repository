package com.tsl.grading_exams;


import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;
import org.apache.commons.lang3.StringUtils;


/**
 * AnInputOutputManager represents the structure for an input output manager with a method to ask about and read
 * a student's name and an exam score.
 * @author Tom Lever
 * @version 1.0
 * @since 05/22/21
 */
class AnInputOutputManager {

	
	/**
	 * scanner is a component of AnInputOutputManager.
	 */
	private static Scanner scanner;
	

	/**
	 * askAboutAndReadAStudentsNameAndAnExamScore prompts a user to propose a student's name and an exam score.
	 * @return
	 */
	protected static ExamPaperStack askAboutAndReadStudentsNamesAndExamScores() throws NoSuchElementException {
		
		ExamPaperStack theExamPaperStack = new ExamPaperStack();
	
		while (true) {
		
			System.out.print("Enter a student's name (or \"end\" to stop): ");
			
			scanner = new Scanner(System.in);
			
			String theProposedStudentsName = scanner.nextLine();
			
			if (theProposedStudentsName.equals("end")) {
				
				if (theExamPaperStack.size() < 1) {
					System.out.println("Exception: The exam-paper stack is empty.");
					continue;
				}
				
				return theExamPaperStack;
			}
			
			if (theProposedStudentsName.equals("") || theProposedStudentsName.equals(" ")) {
				System.out.println("Exception: Invalid name: The proposed name is an empty string or a space.");
				continue;
			}
			
			if (!StringUtils.isAlphaSpace(theProposedStudentsName)) {
				System.out.println("Exception: Invalid students name: Only Unicode letters and spaces are allowed.");
				continue;
			}
			
			
			System.out.print("Enter the exam score for " + theProposedStudentsName + ": ");
			
			scanner = new Scanner(System.in);
			
			int theProposedExamScore;
			try {
				theProposedExamScore = scanner.nextInt();
			}
			catch (InputMismatchException theInputMismatchException) {
				System.out.println("Input mismatch exeption.");
				continue;
			}
			
			if ((theProposedExamScore < 0) || (theProposedExamScore > 100)) {
				System.out.println("Exception: The proposed exam score is outside of the interval 0 to 100.");
				continue;
			}
			
			try {
				theExamPaperStack.checkForDuplicateStudentNamePushAndAddToSumTheScoreOf(
					new ExamPaper(theProposedStudentsName, theProposedExamScore)
				);
			}
			
			catch (ADuplicateStudentNameException theDuplicateStudentNamesException) {
				System.out.println("Exception: A duplicate student name.");
				continue;
			}
			
			catch (AnIntegerOverflowException theIntegerOverflowException) {
				System.out.println(
					"Exception: The sum of the scores in the exam-paper stack is too high for the proposed exam score " +
					theProposedExamScore +
					" to be added to it."
				);
				continue;
			}
			
		}
		
	}
	
}