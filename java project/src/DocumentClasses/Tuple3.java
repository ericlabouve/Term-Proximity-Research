package DocumentClasses;

/**
 * Created by Eric on 2/9/18.
 *
 * @author ericlabouve
 */
public class Tuple3<X, Y, Z> {
    public X x;
    public Y y;
    public Z z;
    public Tuple3(X x, Y y, Z z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    @Override
    public boolean equals(Object _other) {
        Tuple3 other = (Tuple3) _other;
        return this.x.equals(other.x) && this.y.equals(other.y) && this.z.equals(other.z);
    }
}
