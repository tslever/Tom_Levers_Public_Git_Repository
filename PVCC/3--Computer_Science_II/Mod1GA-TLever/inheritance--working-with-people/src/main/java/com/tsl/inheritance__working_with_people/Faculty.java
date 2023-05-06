package com.tsl.inheritance__working_with_people;


/**
 * Faculty represents the structure for a faculty member.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 *
 */
class Faculty extends Employee {

	
	/**
	 * rank is an attribute of a faculty member.
	 */
	private String rank;
	
	
	/**
	 * Faculty() is a zero-argument constructor for Faculty that calls the two-argument constructor of Employee
	 * with arguments "" and 0 and initializes rank to "".
	 */
	public Faculty() {
		//this.name = "";
		//this.empID = 0;
		super("", 0);
		this.rank = "";
	}
	
	
	/**
	 * Faculty(String name, int empID, String rank) is a three-argument constructor for Faculty that calls the
	 * two-argument constructor of Employee with arguments name and empID and initializes rank to rank.
	 * @param name
	 * @param empID
	 * @param rank
	 */
	public Faculty(String name, int empID, String rank) {
		//this.name = name;
		//this.empID = empID;
		super(name, empID);
		this.rank = rank;
	}
	
	
	
	/**
	 * getRank provides this faculty member's rank.
	 * @return
	 */
	public String getRank() {
		return this.rank;
	}
	
	
	/**
	 * setRank sets this faculty member's rank.
	 * @param rank
	 */
	public void setRank(String rank) {
		this.rank = rank;
	}
	
	
	/**
	 * toString provides information for this faculty member.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"Employee ID: " + this.empID + "\n" +
			"Faculty Rank: " + this.rank;
	}
	
}
