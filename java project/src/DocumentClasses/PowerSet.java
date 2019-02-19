/**
 * https://stackoverflow.com/questions/15483036/java-generating-power-set-with-subset-length-between-3-and-a-max
 */

package DocumentClasses;

import java.util.BitSet;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

public class PowerSet<E> implements Iterator<Set<E>>, Iterable<Set<E>> {
    private int     minSize;
    private int     maxSize;
    private E[]     arr     = null;
    private BitSet  bset    = null;

    @SuppressWarnings("unchecked")
    public PowerSet(Set<E> set, int minSize, int maxSize) {
        this.minSize = Math.min(minSize, set.size());
        this.maxSize = Math.min(maxSize, set.size());

        arr = (E[]) set.toArray();
        bset = new BitSet(arr.length + 1);

        for (int i = 0; i < minSize; i++) {
            bset.set(i);
        }
    }

    @Override
    public boolean hasNext() {
        return !bset.get(arr.length);
    }

    @Override
    public Set<E> next() {
        Set<E> returnSet = new TreeSet<E>();
        // System.out.println(printBitSet());
        for (int i = 0; i < arr.length; i++) {
            if (bset.get(i)) {
                returnSet.add(arr[i]);
            }
        }

        int count;
        do {
            incrementBitSet();
            count = countBitSet();
        } while ((count < minSize) || (count > maxSize));

        // System.out.println(returnSet);
        return returnSet;
    }

    protected void incrementBitSet() {
        for (int i = 0; i < bset.size(); i++) {
            if (!bset.get(i)) {
                bset.set(i);
                break;
            } else
                bset.clear(i);
        }
    }

    protected int countBitSet() {
        int count = 0;
        for (int i = 0; i < bset.size(); i++) {
            if (bset.get(i)) {
                count++;
            }
        }
        return count;

    }

    protected String printBitSet() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < bset.size(); i++) {
            if (bset.get(i)) {
                builder.append('1');
            } else {
                builder.append('0');
            }
        }
        return builder.toString();
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not Supported!");
    }

    @Override
    public Iterator<Set<E>> iterator() {
        return this;
    }

    public static void main(String[] args) {
        Set<Character> set = new TreeSet<Character>();
        for (int i = 0; i < 7; i++)
            set.add((char) (i + 'A'));

        PowerSet<Character> pset = new PowerSet<Character>(set, 3, 5);
        int count = 1;
        for (Set<Character> s : pset) {
            System.out.println(count++ + ": " + s);
        }
    }

}
