package com.tsl.grading_exams;



/**
 * ExamPaper represents the structure for an exam paper.
 * @author Tom Lever
 * @version 1.0
 * @since 05/28/21
 *
 */

class ExamPaper {

	
	/**
	 * theStudentsName is an attribute of an exam paper.
	 */
	private String studentsName;
	
	
	/**
	 * theExamScore is an attribute of an exam paper.
	 */
	private int examScore;
	
	
	/**
	 * ExamPaper(String theStudentsNameToUse, int theExamScoreToUse) is the two-parameter constructor for ExamPaper,
	 * which sets this exam paper's studentsName to theStudentsNameToUse and sets examScore to theExamScoreToUse.
	 * @param theStudentsNameToUse
	 * @param theExamScoreToUse
	 */
	protected ExamPaper(String theStudentsNameToUse, int theExamScoreToUse) {
		this.studentsName = theStudentsNameToUse;
		this.examScore = theExamScoreToUse;
	}
	
	
	/**
	 * setTheStudentsName sets this exam paper's student's name.
	 * @param theStudentsNameToUse
	 */
	protected void setTheStudentsName(String theStudentsNameToUse) {
		this.studentsName = theStudentsNameToUse;
	}
	
	
	/**
	 * setTheExamScore sets this exam paper's exam score.
	 * @param theExamScoreToUse
	 */
	protected void setTheExamScore(int theExamScoreToUse) {
		this.examScore = theExamScoreToUse;
	}
	
	
	/**
	 * getTheStudentsName provides this exam paper's student's name.
	 * @return
	 */
	protected String getTheStudentsName() {
		return this.studentsName;
	}
	
	
	/**
	 * getTheExamScore provides this exam paper's exam score.
	 * @return
	 */
	protected int getTheExamScore() {
		return this.examScore;
	}
	
	
	/**
	 * toString provides a string representation of this exam paper.
	 */
	@Override
	public String toString() {
		
		return this.studentsName + "\tScore: " + this.examScore;
		
	}
	
}
