package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_planeswalker extends a_permanent {
	
	private int Combat_Damage;
	private int Loyalty;
	
	public a_planeswalker(String The_Name_To_Use, a_player The_Player_To_Use) {
		super(The_Name_To_Use, The_Player_To_Use);
	}
	
	public void receives_combat_damage(int The_Combat_Damage) {
		this.Combat_Damage = The_Combat_Damage;
	}
}