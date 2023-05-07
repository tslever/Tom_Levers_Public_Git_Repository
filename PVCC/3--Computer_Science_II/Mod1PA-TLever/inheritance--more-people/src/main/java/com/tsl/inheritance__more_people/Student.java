package com.tsl.inheritance__more_people;


/**
 * Student represents the structure for a student.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class Student extends Person implements Payroll {

	/**
	 * studentNumber is an attribute of a student.
	 */
	protected int studentNumber;

	
	/**
	 * Student() is a zero-argument constructor for Student that calls the one-argument constructor of Person with
	 * argument "" and initializes a student's student number to 0.
	 */
	public Student() {
		//this.name = "";
		super("");
		this.studentNumber = 0;
	}
	
	
	/**
	 * Student(String name, int studentNumber) is a two-arguemtn constructor for Student that calls the one-argument
	 * constructor of Person with argument name and initializes a student's student number to studentNumber.
	 * @param name
	 * @param studentNumber
	 */
	public Student(String name, int studentNumber) {
		//this.name = name;
		super(name);
		this.studentNumber = studentNumber;
	}
	
	
	/**
	 * getStudentNumber provides this student's student number.
	 * @return
	 */
	public int getStudentNumber() {
		return this.studentNumber;
	}
	
	
	/**
	 * setStudentNumber sets this student's student number.
	 * @param studentID
	 */
	public void setStudentNumber(int studentID) {
		this.studentNumber = studentID;
	}
	
	
	/**
	 * toString provides information for this student.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"StudentNumber: " + this.studentNumber;
	}
	
}