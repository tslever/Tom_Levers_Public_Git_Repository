package com.tsl.inheritance__more_people;


/**
 * AdjunctFaculty represents the structure for an AdjunctFaculty faculty member.
 * @author Tom Lever
 * @version 1.0
 * @since 05/21/21
 *
 */
public class AdjunctFaculty extends Faculty implements Payroll {

	
	/**
	 * perCreditRate is an attribute of AdjunctFaculty.
	 */
	private double perCreditRate;
	
	
	/**
	 * AdjunctFaculty() is a zero-argument constructor for AdjunctFaculty that calls the three-argument constructor of
	 * Faculty with arguments "", 0, and "" and initializes perCreditRate to 0.00.
	 */
	public AdjunctFaculty() {
		//this.name = "";
		//this.empID = 0;
		//this.rank = "";
		super("", 0, "");
		this.perCreditRate = 0.00;
	}
	
	
	/**
	 * AdjunctFaculty(String name, int empID, String rank, double perCreditRate) is a four-argument constructor for
	 * AdjunctFaculty that calls the three-argument constructor of AdjunctFaculty with arguments name, empID, and rank
	 * and initializes perCreditRate to thePerCreditRateToUse.
	 * @param name
	 * @param empID
	 * @param rank
	 * @param thePerCreditRateToUse
	 */
	public AdjunctFaculty(String name, int empID, String rank, double thePerCreditRateToUse) {
		//this.name = name;
		//this.empID = empID;
		//this.perCreditRate = thePerCreditRateToUse;
		super(name, empID, rank);
		this.perCreditRate = thePerCreditRateToUse;
	}
	
	
	/**
	 * getThePerCreditRate provides this faculty member's per-credit rate.
	 * @return
	 */
	public double getThePerCreditRate() {
		return this.perCreditRate;
	}
	
	
	/**
	 * setThePerCreditRate sets this faculty member's perCreditRate.
	 * @param thePerCreditRateToUse
	 */
	public void setThePerCreditRate(double thePerCreditRateToUse) {
		this.perCreditRate = thePerCreditRateToUse;
	}
	
	
	/**
	 * calculateMonthlyGrossPay() overrides and sufficiently implements calculateMonthlyGrossPay() insufficiently
	 * implemented in Person, and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay() {
		
		return
			this.perCreditRate /*dollars per credit*/ *
			15.0 /*credits per semester*/ *
			1.0 / 2.0 /*semesters per month*/
		;
		
	}
	
	
	/**
	 * calculateMonthlyGrossPay(int creditsPerSemester) overrides and sufficiently implements
	 * calculateMonthlyGrossPay(int creditsPerSemester) insufficiently implemented in Person, and sufficiently
	 * implements the appropriate requirements in Payroll.
	 */
	@Override
	public double calculateMonthlyGrossPay(int creditsPerSemester) {
		
		return
			this.perCreditRate /*dollars per credit*/ *
			(double)creditsPerSemester /*credits per semester*/ *
			1.0 / 2.0 /*semesters per month*/
		;
		
	}
	
	
	/**
	 * displayPayInfo() overrides and sufficiently implements displayPayInfo() insufficiently implemented in Person,
	 * and sufficiently implements the appropriate requirements in Payroll.
	 */
	@Override
	public void displayPayInfo() {
		
		int creditsPerSemester = 12;
		
		System.out.println(
			"Pay is done twice per semester.\n" +
			"Pay per credit: " + this.perCreditRate + "\n" +
			"Period pay: " + calculateMonthlyGrossPay(creditsPerSemester) + "\n"
		);
		
	}
	
	
}