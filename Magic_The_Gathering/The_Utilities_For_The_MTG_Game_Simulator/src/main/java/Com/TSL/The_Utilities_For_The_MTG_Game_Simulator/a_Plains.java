package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_Plains extends a_land {

	public a_Plains(boolean The_Tapped_Status) {
		super("Plains", The_Tapped_Status);
	}
	
	@Override
	public a_mana_pool indicates_mana_pool_it_would_contribute_for(int The_Option_For_Contributing_Mana_To_Use) {
		return new a_mana_pool(0, 0, 0, 0, 0, 1);
	}
	
	@Override
	public a_mana_pool provides_a_mana_pool_for(int The_Option_For_Contributing_Mana_To_Use) {
		this.taps();
		return new a_mana_pool(0, 0, 0, 0, 0, 1);
	}
}
