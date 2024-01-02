package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class an_ability {
	
	private String Effect;
	private a_permanent Permanent;
	
	public an_ability(String The_Effect_To_Use, a_permanent The_Permanent_To_Use) {
		this.Effect = The_Effect_To_Use;
		this.Permanent = The_Permanent_To_Use;
	}

	public String provides_its_effect() {
		return this.Effect;
	}
	
	public a_permanent provides_its_permanent() {
		return this.Permanent;
	}
	
	@Override
	public String toString() {
		return this.Permanent.provides_its_name() + ": " + this.Effect;
	}
	
}
