package com.tsl.inheritance__more_people;


/**
 * Staff represents the structure for a staff member.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
class Staff extends Employee implements Payroll {

	
	/**
	 * rank is an attribute of a staff member.
	 */
	protected String rank;
	
	
	/**
	 * Staff() is a zero-argument constructor for Staff that calls the two-argument constructor of Employee
	 * with arguments "" and 0 and initializes rank to "".
	 */
	public Staff() {
		//this.name = "";
		//this.empID = 0;
		super("", 0);
		this.rank = "";
	}
	
	
	/**
	 * Staff(String name, int empID, String rank) is a three-argument constructor for Staff that calls the
	 * two-argument constructor of Employee with arguments name and empID and initializes rank to rank.
	 * @param name
	 * @param empID
	 * @param rank
	 */
	public Staff(String name, int empID, String rank) {
		//this.name = name;
		//this.empID = empID;
		super(name, empID);
		this.rank = rank;
	}
	
	
	
	/**
	 * getRank provides this staff member's rank.
	 * @return
	 */
	public String getRank() {
		return this.rank;
	}
	
	
	/**
	 * setRank sets this staff member's rank.
	 * @param rank
	 */
	public void setRank(String rank) {
		this.rank = rank;
	}
	
	
	/**
	 * toString provides information for this staff member.
	 */
	@Override
	public String toString() {
		return
			"Name: " + this.name + "\n" +
			"Employee ID: " + this.empID + "\n" +
			"Staff Rank: " + this.rank;
	}
	
}