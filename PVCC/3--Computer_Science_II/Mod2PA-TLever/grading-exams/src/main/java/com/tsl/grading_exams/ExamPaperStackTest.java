package com.tsl.grading_exams;


import org.junit.jupiter.api.Test;


public class ExamPaperStackTest {

	
	private LLNode<ExamPaper> top;
	
	private int sumOfTheExamScores = 100;
	
	
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
	
	
	// See CheckTheAdditionOfTest for tests of checkTheAdditionOf.
	
	
	@Test
	public void testGetTheAverageExamScoreForOneExam() {
		
		System.out.println("Running testGetTheAverageExamScoreForOneExam.");
		
		try {

			this.top = new LLNode<ExamPaper>(new ExamPaper("Tom Lever", 100));
			
			if (size() < 1) {
				throw new ANoExamScoresExistToAverageException("Exception: No exam scores exist to average.");
			}
			
			System.out.println("The average exam score: " + (double)this.sumOfTheExamScores / (double)size());
			
		}
		
		catch (ANoExamScoresExistToAverageException theNoExamScoresExistToAverageException) {
			System.out.println("Exception: No exam scores exist to average.");
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testGetTheAverageExamScoreForZeroExams() {
		
		System.out.println("Running testGetTheAverageExamScoreForZeroExams.");
		
		try {

			this.top = null;
			
			if (size() < 1) {
				throw new ANoExamScoresExistToAverageException("Exception: No exam scores exist to average.");
			}
			
			System.out.println("The average exam score: " + (double)this.sumOfTheExamScores / (double)size());
			
		}
		
		catch (ANoExamScoresExistToAverageException theNoExamScoresExistToAverageException) {
			System.out.println("Exception: No exam scores exist to average.");
		}
		
		System.out.println();
		
	}
	
	
	private boolean isEmpty() {
	// Returns true if this stack is empty, otherwise returns false.

		return (top == null);
		
	}

		
	@Test
	public void testPopForOneExam() {
		
		System.out.println("Running testPopForOneExam.");
		
		this.top = new LLNode<ExamPaper>(new ExamPaper("Tom Lever", 100));
		
		try {
		
			if (isEmpty()) {
				throw new StackUnderflowException("Pop attempted on an empty stack.");
			}
			
			top = top.getTheReferenceToTheNextLinkedListNode();
			
			System.out.println("Popped a linked-list node containing an exam paper from an exam-paper stack.");
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			
			System.out.println(theStackUnderflowException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testPopForZeroExams() {
		
		System.out.println("Running testPopForZeroExams.");
		
		try {
		
			if (isEmpty()) {
				throw new StackUnderflowException("Pop attempted on an empty stack.");
			}
			
			top = top.getTheReferenceToTheNextLinkedListNode();
			
			System.out.println("Popped a linked-list node containing an exam paper from an exam-paper stack.");
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			
			System.out.println(theStackUnderflowException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testTopForOneExam() {
		
		System.out.println("Running testTopForOneExam.");
		
		this.top = new LLNode<ExamPaper>(new ExamPaper("Tom Lever", 100));
		
		try {
		
			if (isEmpty()) {
				throw new StackUnderflowException("Pop attempted on an empty stack.");
			}
			
			System.out.println(this.top.getTheInformation());
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			
			System.out.println(theStackUnderflowException.getMessage());
			
		}
		
		System.out.println();
		
	}

	
	@Test
	public void testTopForZeroExams() {
		
		System.out.println("Running testTopForZeroExams.");
		
		try {
		
			if (isEmpty()) {
				throw new StackUnderflowException("Pop attempted on an empty stack.");
			}
			
			System.out.println(this.top.getTheInformation());
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			
			System.out.println(theStackUnderflowException.getMessage());
			
		}
		
		System.out.println();
		
	}
	
	
	public void pop() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise removes top element from this stack.

		if (isEmpty()) {
			throw new StackUnderflowException("Pop attempted on an empty stack.");
		}
		
		top = top.getTheReferenceToTheNextLinkedListNode();
		
	}
	
	
	@Test
	public void testPopSomeForOneExam() {
		
		System.out.println("Running testPopSomeForOneExam.");
		
		this.top = new LLNode<ExamPaper>(new ExamPaper("Tom Lever", 100));
		int count = 1;
		
		try {
			
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
			
			System.out.println("Popped one exam paper from an exam-paper stack.");
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testPopSomeForZeroExams() {
		
		System.out.println("Running testPopSomeForZeroExams.");
		
		int count = 1;
		
		try {
			
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
			
			System.out.println("Popped one exam paper from an exam-paper stack.");
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	private ExamPaper top() {
	// Throws StackUnderflowException if this stack is empty,
	// otherwise returns the information of the top element of this stack.

		if (isEmpty()) {
			throw new StackUnderflowException("Top attempted on an empty stack.");
		}
		
		return top.getTheInformation();
	}
	
	
	@Test
	public void testPopTopForOneExam() {
		
		System.out.println("Running testPopTopForOneExam.");
		
		this.top = new LLNode<ExamPaper>(new ExamPaper("Tom Lever", 100));
		
		try {
			
			if (isEmpty()) {
				throw new StackUnderflowException(
					"Exception: poptop for an improved array-based bounded stack was requested when the stack was empty."
				);
			}
			
			ExamPaper theInformationOfTheTopLinkedListNode = top();
			pop();
			
			System.out.println(theInformationOfTheTopLinkedListNode);
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
	@Test
	public void testPopTopForZeroExams() {
		
		System.out.println("Running testPopTopForZeroExams.");
		
		try {
			
			if (isEmpty()) {
				throw new StackUnderflowException(
					"Exception: poptop for an improved array-based bounded stack was requested when the stack was empty."
				);
			}
			
			ExamPaper theInformationOfTheTopLinkedListNode = top();
			pop();
			
			System.out.println(theInformationOfTheTopLinkedListNode);
		
		}
		
		catch (StackUnderflowException theStackUnderflowException) {
			System.out.println(theStackUnderflowException.getMessage());
		}
		
		System.out.println();
		
	}
	
	
}
