package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class an_activated_ability {

	private String Cost;
	private String Effect;
	private a_permanent Permanent;
	
	public an_activated_ability(String The_Cost_To_Use, String The_Effect_To_Use, a_permanent The_Permanent_To_Use) {
		this.Cost = The_Cost_To_Use;
		this.Effect = The_Effect_To_Use;
		this.Permanent = The_Permanent_To_Use;
	}
	
	public String provides_its_cost() {
		return this.Cost;
	}
	
	public String provides_its_effect() {
		return this.Effect;
	}
	
	public a_permanent provides_its_permanent() {
		return this.Permanent;
	}
	
	public boolean requires_tapping() {
		return true;
	}
	
}
