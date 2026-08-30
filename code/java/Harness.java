// Generic JSON marshalling harness for the translated program.
//
// One harness serves every problem: the Java method under test is located by reflection from a
// type descriptor passed on the command line, so no per-problem glue is generated and no
// per-problem glue can differ between arms. Reads one JSON argument array per stdin line and
// writes one JSON result object per line, flushed immediately so a hang loses only the tail.
//
// Deliberately dependency-free: Java 17 ships no JSON parser, and adding one would put a
// third-party library between the measurement and the result.
//
// Result lines: {"ok":true,"v":<value>} or {"ok":false,"e":"<ExceptionSimpleName>"}
// Non-finite doubles are emitted as bare Infinity / -Infinity / NaN, which Python's json module
// accepts by default; that keeps a legitimately infinite return value distinguishable from an
// error rather than collapsing both into a failure.

import java.io.*;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

public class Harness {

    static final int MAX_RESULT_CHARS = 1_000_000;

    // ------------------------------------------------------------------ minimal JSON reader
    static final class P {
        final String s; int i = 0;
        P(String s) { this.s = s; }
        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }
        char peek() { ws(); return i < s.length() ? s.charAt(i) : '\0'; }
        Object value() {
            ws();
            char c = peek();
            if (c == '[') return array();
            if (c == '"') return str();
            if (c == 't') { expect("true"); return Boolean.TRUE; }
            if (c == 'f') { expect("false"); return Boolean.FALSE; }
            if (c == 'n') { expect("null"); return null; }
            return number();
        }
        void expect(String w) {
            if (!s.startsWith(w, i)) throw new RuntimeException("bad literal at " + i);
            i += w.length();
        }
        List<Object> array() {
            List<Object> out = new ArrayList<>();
            ws(); i++;                       // consume '['
            if (peek() == ']') { i++; return out; }
            while (true) {
                out.add(value());
                char c = peek();
                if (c == ',') { i++; continue; }
                if (c == ']') { i++; return out; }
                throw new RuntimeException("bad array at " + i);
            }
        }
        String str() {
            ws(); i++;                       // consume opening quote
            StringBuilder b = new StringBuilder();
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c != '\\') { b.append(c); continue; }
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': b.append('\n'); break;
                    case 't': b.append('\t'); break;
                    case 'r': b.append('\r'); break;
                    case 'b': b.append('\b'); break;
                    case 'f': b.append('\f'); break;
                    case '/': b.append('/');  break;
                    case '"': b.append('"');  break;
                    case '\\': b.append('\\'); break;
                    case 'u':
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default: throw new RuntimeException("bad escape " + e);
                }
            }
        }
        Object number() {
            ws();
            int st = i;
            if (s.startsWith("Infinity", i))  { i += 8; return Double.POSITIVE_INFINITY; }
            if (s.startsWith("-Infinity", i)) { i += 9; return Double.NEGATIVE_INFINITY; }
            if (s.startsWith("NaN", i))       { i += 3; return Double.NaN; }
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            String t = s.substring(st, i);
            if (t.indexOf('.') < 0 && t.indexOf('e') < 0 && t.indexOf('E') < 0) {
                try { return Long.parseLong(t); }
                catch (NumberFormatException ex) { return Double.parseDouble(t); }
            }
            return Double.parseDouble(t);
        }
    }

    // ------------------------------------------------------------------ minimal JSON writer
    static void write(StringBuilder b, Object o) {
        if (o == null) { b.append("null"); return; }
        if (o instanceof String s) { quote(b, s); return; }
        if (o instanceof Boolean bo) { b.append(bo ? "true" : "false"); return; }
        if (o instanceof Double d) {
            if (d.isNaN()) b.append("NaN");
            else if (d.isInfinite()) b.append(d > 0 ? "Infinity" : "-Infinity");
            else b.append(d.toString());
            return;
        }
        if (o instanceof Float f) { write(b, (double) f); return; }
        if (o instanceof Number n) { b.append(n.toString()); return; }
        if (o instanceof Character c) { quote(b, c.toString()); return; }
        if (o.getClass().isArray()) {
            b.append('[');
            int n = Array.getLength(o);
            for (int k = 0; k < n; k++) { if (k > 0) b.append(','); write(b, Array.get(o, k)); }
            b.append(']');
            return;
        }
        if (o instanceof Collection<?> c) {
            b.append('[');
            boolean first = true;
            for (Object x : c) { if (!first) b.append(','); write(b, x); first = false; }
            b.append(']');
            return;
        }
        quote(b, o.toString());
    }

    static void quote(StringBuilder b, String s) {
        b.append('"');
        for (int k = 0; k < s.length(); k++) {
            char c = s.charAt(k);
            switch (c) {
                case '"':  b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n");  break;
                case '\r': b.append("\\r");  break;
                case '\t': b.append("\\t");  break;
                case '\b': b.append("\\b");  break;
                case '\f': b.append("\\f");  break;
                default:
                    if (c < 0x20 || c > 0x7e) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        b.append('"');
    }

    // ------------------------------------------------------------------ type coercion
    static Class<?> classFor(String tag) {
        switch (tag) {
            case "long":      return long.class;
            case "double":    return double.class;
            case "boolean":   return boolean.class;
            case "String":    return String.class;
            case "long[]":    return long[].class;
            case "double[]":  return double[].class;
            case "boolean[]": return boolean[].class;
            case "String[]":  return String[].class;
            case "long[][]":  return long[][].class;
            case "double[][]":return double[][].class;
            case "String[][]":return String[][].class;
        }
        throw new RuntimeException("unknown type tag " + tag);
    }

    @SuppressWarnings("unchecked")
    static Object coerce(Object v, String tag) {
        switch (tag) {
            case "long":
                if (v instanceof Long l) return l;
                if (v instanceof Boolean b) return b ? 1L : 0L;
                if (v instanceof Double d) {
                    if (d != Math.rint(d) || d.isInfinite() || d.isNaN())
                        throw new IllegalArgumentException("non-integral for long");
                    return d.longValue();
                }
                break;
            case "double":
                if (v instanceof Double d) return d;
                if (v instanceof Long l) return l.doubleValue();
                if (v instanceof Boolean b) return b ? 1.0 : 0.0;
                break;
            case "boolean":
                if (v instanceof Boolean b) return b;
                break;
            case "String":
                if (v instanceof String s) return s;
                break;
            default:
                if (tag.endsWith("[]")) {
                    String inner = tag.substring(0, tag.length() - 2);
                    List<Object> src = (List<Object>) v;
                    Object arr = Array.newInstance(classFor(inner), src.size());
                    for (int k = 0; k < src.size(); k++)
                        Array.set(arr, k, coerce(src.get(k), inner));
                    return arr;
                }
        }
        throw new IllegalArgumentException("cannot coerce " + v + " to " + tag);
    }

    // ------------------------------------------------------------------ main loop
    public static void main(String[] argv) throws Exception {
        String method = argv[0];
        String[] ptags = argv[1].isEmpty() ? new String[0] : argv[1].split(",");
        long perCallMs = Long.parseLong(argv[2]);
        int maxTimeouts = Integer.parseInt(argv[3]);

        Class<?>[] ptypes = new Class<?>[ptags.length];
        for (int k = 0; k < ptags.length; k++) ptypes[k] = classFor(ptags[k]);

        Class<?> sol = Class.forName("Solution");
        Method m = sol.getMethod(method, ptypes);
        m.setAccessible(true);

        BufferedReader in = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out),
                true, StandardCharsets.UTF_8);
        ExecutorService pool = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r); t.setDaemon(true); return t;
        });

        int timeouts = 0;
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            StringBuilder b = new StringBuilder();
            try {
                List<Object> raw = (List<Object>) new P(line).value();
                Object[] args = new Object[ptags.length];
                for (int k = 0; k < ptags.length; k++) args[k] = coerce(raw.get(k), ptags[k]);
                Future<Object> f = pool.submit(() -> m.invoke(null, args));
                Object r;
                try {
                    r = f.get(perCallMs, TimeUnit.MILLISECONDS);
                } catch (TimeoutException te) {
                    f.cancel(true);
                    timeouts++;
                    out.println("{\"ok\":false,\"e\":\"Timeout\"}");
                    if (timeouts >= maxTimeouts) {
                        // exit reporting ONLY the timeout, so the driver sees a trailing Timeout
                        // line and resumes from the input after it. A cancelled Future's thread
                        // keeps running, so the JVM must die rather than carry it forward.
                        out.flush();
                        Runtime.getRuntime().halt(3);
                    }
                    continue;
                }
                b.append("{\"ok\":true,\"v\":");
                write(b, r);
                b.append('}');
                // Symmetric with the Python side: a result too large to serialise is treated as
                // resource exhaustion, not as a value, so it cannot become a false divergence.
                if (b.length() > MAX_RESULT_CHARS) {
                    b.setLength(0);
                    b.append("{\"ok\":false,\"e\":\"Oversize\"}");
                }
            } catch (ExecutionException ee) {
                Throwable c = ee.getCause();
                if (c instanceof InvocationTargetException ite && ite.getCause() != null)
                    c = ite.getCause();
                b.setLength(0);
                b.append("{\"ok\":false,\"e\":");
                quote(b, c.getClass().getSimpleName());
                b.append('}');
            } catch (Throwable t) {
                Throwable c = (t instanceof InvocationTargetException ite && ite.getCause() != null)
                        ? ite.getCause() : t;
                b.setLength(0);
                b.append("{\"ok\":false,\"e\":");
                quote(b, c.getClass().getSimpleName());
                b.append('}');
            }
            out.println(b);
        }
        out.flush();
        System.exit(0);
    }
}
