package Com.TSL.MirrorCollectionUtilities;


public class CollectionT<T> implements CollectionInterface<T> {
	private T[] data;
	private int count = 0;

	public CollectionT(int var1) {
		this.data = (T[]) (new Object[var1]);
	}

	public void add(T var1) {
		if (this.isFull()) {
			this.doubling();
		}

		this.data[this.count++] = var1;
	}

	public void add(T var1, int var2) {
		if (var2 >= 0) {
			if (var2 > this.count) {
				var2 = this.count;
			}

			if (this.isFull()) {
				this.doubling();
			}

			for (int var3 = this.count - 1; var3 >= var2; --var3) {
				this.data[var3 + 1] = this.data[var3];
			}

			this.data[var2] = var1;
			++this.count;
		}
	}

	public T get(T var1) {
		int var2 = this.indexOf(var1);
		return var2 >= 0 ? this.data[var2] : null;
	}

	public T get(int var1) {
		return var1 >= 0 && var1 <= this.count - 1 ? this.data[var1] : null;
	}

	public boolean isFull() {
		return this.count == this.data.length;
	}

	public boolean isEmpty() {
		return this.count == 0;
	}

	public int size() {
		return this.count;
	}

	public int indexOf(T var1) {
		for (int var2 = 0; var2 < this.count; ++var2) {
			if (this.data[var2].equals(var1)) {
				return var2;
			}
		}

		return -1;
	}

	public boolean contains(T var1) {
		return this.indexOf(var1) != -1;
	}

	public boolean remove(int var1) {
		if (var1 >= 0 && var1 < this.count) {
			this.data[var1] = this.data[this.count - 1];
			--this.count;
			return true;
		} else {
			return false;
		}
	}

	public int remove(T var1) {
		int var2 = 0;

		for (int var3 = this.indexOf(var1); var3 != -1; var3 = this.indexOf(var1)) {
			this.remove(var3);
			++var2;
		}

		return var2;
	}

	public void print() {
		String var1 = "";

		for (int var2 = 0; var2 < this.count; ++var2) {
			var1 = var1 + this.data[var2].toString() + ", ";
		}

		if (var1.length() > 0) {
			var1 = var1.substring(0, var1.length() - 2);
		}

		System.out.println("[" + var1 + "]");
	}

	private void doubling() {
		T[] var1 = (T[]) (new Object[this.data.length * 2]);

		for (int var2 = 0; var2 < this.data.length; ++var2) {
			var1[var2] = this.data[var2];
		}

		this.data = var1;
	}
}