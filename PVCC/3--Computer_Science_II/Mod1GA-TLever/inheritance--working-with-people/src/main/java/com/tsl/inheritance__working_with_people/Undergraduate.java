package com.tsl.inheritance__working_with_people;


/**
 * Undergraduate represents the structure for an undergraduate.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 *
 */
class Undergraduate extends Student {

	/**
	 * major is an attribute of an undergraduate.
	 */
	private String major;
	
	
	/**
	 * Undergraduate() is a zero-argument constructor for Undergraduate that calls the two-argument constructor
	 * of Student with arguments "" and 0 and initializes major to "".
	 */
	public Undergraduate() {
		//this.name = "";
		//this.studentNumber = 0;
		super("", 0);
		this.major = "";
	}
	
	
	/**
	 * Undergraduate(String name, int studentNumber, String major) is a three-argument constructor for Undergraduate
	 * that calls the two-argument constructor of Student with arguments name and studentNumber and initializes
	 * major to major.
	 * @param name
	 * @param studentNumber
	 * @param major
	 */
	public Undergraduate(String name, int studentNumber, String major) {
		//this.name = name;
		//this.studentNumber = studentNumber;
		super(name, studentNumber);
		this.major = major;
	}
	
	
	/**
	 * getMajor provides this undergraduate's major.
	 * @return
	 */
	public String getMajor() {
		return this.major;
	}
	
	
	/**
	 * setMajor sets this undergraduate's major.
	 * @param major
	 */
	public void setMajor(String major) {
		this.major = major;
	}
	
	
	/**
	 * toString provides information for this undergraduate.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"StudentNumber: " + this.studentNumber + "\n" +
			"Undergraduate major: " + this.major;
	}
	
}
