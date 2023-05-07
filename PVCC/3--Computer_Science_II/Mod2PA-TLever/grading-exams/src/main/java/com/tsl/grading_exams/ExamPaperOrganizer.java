package com.tsl.grading_exams;


/**
 * ExamPaperOrganizer encapsulates the entry point of this program that requests a user to enter students' names and
 * scores on an exam. The program creates a exam-paper stack with exam papers corresponding to the students. The program
 * outputs a summary of and statistics relating to the exam-paper stack. The program distributes the exam papers in the
 * stack of all exam papers to a stack of exam papers with scores less than the average exam score and to a stack of
 * exam papers with scores greater than or equal to the average exam score. The program outputs summaries of and
 * statistics relating to each of the latest stacks.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/29/21
 *
 */
public class ExamPaperOrganizer {

	
	/**
	 * main is the entry point of this program that outputs summaries and statistics relating to a stack of exam papers
	 * with scores less than the average score for the exam and to a stack of exam papers with scores greater than or
	 * equal to the average score. A no exam scores exist to average exception is caught if data entry was terminated
	 * before a exam paper was added to the exam paper stack; a message is provided and data entry begins again.
	 * 
	 * @param args
	 */
    public static void main( String[] args ) throws ANoExamScoresExistToAverageException {

    	
    	ExamPaperStack theExamPaperStackForAllExams =
    		AnInputOutputManager.askAboutAndReadStudentsNamesAndExamScores();
    	System.out.println(
    		"\nThe exam-paper stack contains papers with the following names and scores: {\n" +
    		theExamPaperStackForAllExams + "\n" +
    		"}"
    	);
    	
    	System.out.println("The number of exam papers is: " + theExamPaperStackForAllExams.size());
    	
    	double theAverageExamScore = theExamPaperStackForAllExams.getTheAverageExamScore();
		System.out.println(
			"The average of the scores for the " + theExamPaperStackForAllExams.size() + " exam paper(s) is: " +
			theAverageExamScore + "\n"
		);
    	
    	ExamPaperStack theExamPaperStackForExamsWithScoresLessThanTheAverage = new ExamPaperStack();
    	ExamPaperStack theExamPaperStackForExamsWithScoresGreaterThanOrEqualToTheAverage = new ExamPaperStack();
    	
    	while (!theExamPaperStackForAllExams.isEmpty()) {
    		
    		ExamPaper theExamPaper = theExamPaperStackForAllExams.poptop();
    		
    		if ((double)theExamPaper.getTheExamScore() < theAverageExamScore) {
    			theExamPaperStackForExamsWithScoresLessThanTheAverage.push(theExamPaper);
    		}
    		else {
    			theExamPaperStackForExamsWithScoresGreaterThanOrEqualToTheAverage.push(theExamPaper);
    		}
    		
    	}
    	
    	System.out.println(
    		"The exam papers with scores greater than or equal to the average exam score of " + theAverageExamScore +
    		" have the following names and scores: {\n" +
    		theExamPaperStackForExamsWithScoresGreaterThanOrEqualToTheAverage + "\n" +
    		"}\n" +
    		"The number of exam papers with scores greater than or equal to the average exam score: " +
    		theExamPaperStackForExamsWithScoresGreaterThanOrEqualToTheAverage.size() + "\n"
    	);
    	
    	System.out.println(
    		"The exam papers with scores less than the average exam score of " + theAverageExamScore +
    		" have the following names and scores: {\n" +
    		theExamPaperStackForExamsWithScoresLessThanTheAverage + "\n" +
    		"}\n" +
    		"The number of exam papers with scores less than the average exam score: " +
    		theExamPaperStackForExamsWithScoresLessThanTheAverage.size() + "\n"
    	);
    	
    }
    
}
