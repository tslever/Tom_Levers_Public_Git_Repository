package com.tsl.grading_exams;


import java.util.ArrayList;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 05/28/21
*/

class ExamPaperStack implements StackInterface<ExamPaper> {

	
	/**
	 * THE_MINIMUM_INTEGER is an attribute of this exam-paper stack.
	 */
	private final int THE_MINIMUM_INTEGER = -2147483648;
	
	
	/**
	 * THE_MAXIMUM_INTEGER is an attribute of this exam-paper stack.
	 */
	private final int THE_MAXIMUM_INTEGER = 2147483647;
	
	

	private LLNode<ExamPaper> top;   // reference to the top of this stack
	
	
	/**
	 * sumOfTheExamScore is an attribute of this exam-paper stack.
	 */
	private int sumOfTheExamScores;
	
	
	/**
	 * theStudentsNames is a component of this exam-paper stack.
	 */
	ArrayList<String> theStudentNames;
	
	
	protected ExamPaperStack() {
	/**
	 * ExamPaperStack() is a zero-parameter constructor for ExamPaperStack that sets the top linked-list node
	 * of this linked list based stack to null, sets sumOfTheExamScores to 0, and sets theStudentsNames to an empty
	 * object of type ArrayList.
	 */
		
		this.top = null;
		this.sumOfTheExamScores = 0;
		this.theStudentNames = new ArrayList<String>();
		
	}
	
	
	/**
	 * checkWhetherIsDuplicate(String theStudentName) throws a duplicate student name exception if this exam-paper
	 * stack's array of students' names contains the student-name argument.
	 * @param theStudentName
	 * @throws ADuplicateStudentNameException
	 */
	private void checkWhetherIsDuplicate(String theStudentName) throws ADuplicateStudentNameException {
			
		if (this.theStudentNames.contains(theStudentName)) {
			throw new ADuplicateStudentNameException("Exception: Duplicate student name.");
		}
		
	}
	
	
	/**
	 * checkTheAdditionOf throws an integer-overflow exception if addition of a first integer and a
	 * second integer would result in a sum greater than the maximum integer or less than the minimum integer.
	 * @param theFirstInteger
	 * @param theSecondInteger
	 * @throws AnIntegerOverflowException
	 */
	private void checkTheAdditionOf(int theFirstInteger, int theSecondInteger) throws AnIntegerOverflowException {
		
		if ((theFirstInteger > 0) && (theSecondInteger > 0) && (theFirstInteger > this.THE_MAXIMUM_INTEGER - theSecondInteger) ||
			(theFirstInteger < 0) && (theSecondInteger < 0) && (theFirstInteger < this.THE_MINIMUM_INTEGER - theSecondInteger)) {
			
			throw new AnIntegerOverflowException(
				"Integer-overflow exception: the sum of " + theFirstInteger + " and " + theSecondInteger +
				" is outside the interval [" + this.THE_MINIMUM_INTEGER + ", " + this.THE_MAXIMUM_INTEGER + "]."
			);
			
		}
		
	}
	
	
	/**
	 * getTheAverageExamScore provides the average of the exam scores for all of the exam papers in this stack, or
	 * throws a no exam scores exist to average exception.
	 * @return
	 * @throws ANoExamScoresExistToAverageException
	 */
	protected double getTheAverageExamScore() throws ANoExamScoresExistToAverageException {
		
		if (size() < 1) {
			throw new ANoExamScoresExistToAverageException("Exception: No exam scores exist to average.");
		}
		
		return (double)this.sumOfTheExamScores / (double)size();
				
	}
	
	
	public void push(ExamPaper theExamPaper) {
	// Places element at the top of this stack.

		LLNode<ExamPaper> newNode = new LLNode<ExamPaper>(theExamPaper);
		newNode.setsItsReferenceTo(top);
		top = newNode;
		
	}
	
	
	/**
	 * checkForDuplicateStudentNamePushAndAddToSumTheScoreOf checks whether the student's name corresponding to its
	 * exam-paper argument is a duplicate (and throws a duplicate student name exception if it is), adds the name to
	 * this exam-paper stack's array of student's names, pushes the exam paper onto this stack, checks the addition of
	 * the running sum of exam scores corresponding to the exam papers on this stack and the present exam paper's score
	 * (and throws an integer-overflow exception if the check fails), and adds the present exam paper's score to the
	 * running sum.
	 * 
	 * @param theExamPaper
	 * @throws ADuplicateStudentNameException
	 * @throws AnIntegerOverflowException
	 */
	public void checkForDuplicateStudentNamePushAndAddToSumTheScoreOf(ExamPaper theExamPaper)
		throws ADuplicateStudentNameException, AnIntegerOverflowException {
		
		checkWhetherIsDuplicate(theExamPaper.getTheStudentsName());
		
		this.theStudentNames.add(theExamPaper.getTheStudentsName());
		
		push(theExamPaper);
		
		checkTheAdditionOf(this.sumOfTheExamScores, theExamPaper.getTheExamScore());
		
		this.sumOfTheExamScores += theExamPaper.getTheExamScore();
		
	}
	

	public void pop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes top element from this stack.

		if (isEmpty()) {
			throw new StackUnderflowException("Pop attempted on an empty stack.");
		}
		
		top = top.getTheReferenceToTheNextLinkedListNode();
		
	}

	
	public ExamPaper top() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise returns the information of the top element of this stack.

		if (isEmpty()) {
			throw new StackUnderflowException("Top attempted on an empty stack.");
		}
		
		return top.getTheInformation();
	}

	
	public boolean isEmpty() {
	// Returns true if this stack is empty, otherwise returns false.

		return (top == null);
	}

	
	public boolean isFull() {
	// Returns false - a linked stack is never full

		return false;
		
	}

		//*** Task #1: define method toString(): String
		//*	creates and returns a string that correctly represents the current stack.


	@Override
	public String toString() {
	/**
	 * toString arranges strings representing the information of the linked-list nodes in this stack in an indented
	 * column.
	 */
		
		String theStringRepresentingTheStack = "";
		
		LLNode<ExamPaper> theCurrentNode = top;
		if (theCurrentNode != null) {
			theCurrentNode = top;
			theStringRepresentingTheStack += "\t" + top.getTheInformation();
			theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		}
		
		while (theCurrentNode != null) {
			theStringRepresentingTheStack += "\n\t" + theCurrentNode.getTheInformation();
			theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		}
		
		return theStringRepresentingTheStack;

	}

		//*** Task #2: define method size(): int
		//*	returns a count of how many items are currently on the stack.


	protected int size() {
	// returns a count of how many elements are on the stack
		
		int theNumberOfElementsOnTheStack = 0;
		
		LLNode<ExamPaper> theCurrentNode = top;
		while (theCurrentNode != null) {
			theNumberOfElementsOnTheStack++;
			theCurrentNode = theCurrentNode.getTheReferenceToTheNextLinkedListNode();
		}
		
		return theNumberOfElementsOnTheStack;

	}

		//*** Task #3: define method  popSome(int count): void
		//*	removes the top count elements from the stack

	
	protected void popSome(int count) {
	// if possible, removes top count elements from stack;
	// otherwise throws StackUnderflowException

		if (size() < count) {
			throw new StackUnderflowException(
				"Exception: popSome(count=" + count + ") for an improved linked list based stack was requested when " +
				"the number of elements on the stack, size = " + size() + ", was less than count = " + count + "."
			);
		}
		
		while (count > 0) {
			pop();
			count--;
		}

	}

		//*** Task #4: define method  swapStart(): boolean
		//*	 if there are less than two elements on the stack returns false;
		//*	 otherwise it reverses the order of the top two elements on the stack and returns true

	protected boolean swapStart() {
	// if possible, reverses order of top 2 elements and returns true;
	// otherwise returns false
		
		if (size() < 2) {
			return false;
		}
		
		LLNode<ExamPaper> theNodeInStorage = top.getTheReferenceToTheNextLinkedListNode();
		top.setsItsReferenceTo(top.getTheReferenceToTheNextLinkedListNode().getTheReferenceToTheNextLinkedListNode());
		theNodeInStorage.setsItsReferenceTo(top);
		top = theNodeInStorage;
		return true;
		
	}
		//*** Task #5: define method poptop( ): T
		//*	 the “classic” pop operation, if the stack is empty it throws StackUnderflowException;
		//*	 otherwise it both removes and returns the top element of the stack.


	protected ExamPaper poptop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes and returns top element from this stack.

		if (isEmpty()) {
			throw new StackUnderflowException(
				"Exception: poptop for an improved array-based bounded stack was requested when the stack was empty."
			);
		}
		
		ExamPaper theInformationOfTheTopLinkedListNode = top();
		pop();
		return theInformationOfTheTopLinkedListNode;

	}

}
