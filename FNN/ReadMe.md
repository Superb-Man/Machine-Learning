\[
(x^{(m)})^{\beta} - 1 \bigg[ b^{(1)} - (x^{(m)})^{\beta} \bigg]
\]

\[
(z_2^{(l-1)}) \log x^{(m)} - (x^{(m)})^{\beta} \log x^{(m)} - x^{\beta} \log x^{(m)} - y^{(l)} \log \left( z^{(l-1)} \right)
\]

\[
z = y + x^{(m)}, w + x^{(m)} \Rightarrow \quad x^{m} = \gamma^{\beta}
\]

---

\[
\frac{dL}{dz} \quad \frac{dL}{dz^{(l)}} \quad \frac{dL}{d^2w}
\]

\[
\frac{dL}{dz^{l-1}} \quad \frac{dL}{dL^{l-1}}
\]

\[
\frac{dL}{d(a^{(l)})} \quad \frac{dL}{d \left( z^{(l)} \right)} \quad \frac{dL}{dw}
\]

\[
a(1-a)
\]

---

\[
\frac{dL}{da^{(l)}} \quad a(1-a)
\]

**For cross-entropy (outer layer)**

---

\[
\frac{dL}{dw}
\]

---

See we will compute it separately in each layer with the same as 1th layer

\[
\frac{dL}{dz^{l-1}} \quad \frac{dL}{d(a^{(l)})} \quad d^2(w^{l})^{T} \quad z^{(l-1)}
\]

---

Actually,

\[
\frac{dL}{dz^{l}} = \frac{dL}{dz^{l-1}} = \frac{dL}{dz^{l}}^{T}
\]

**Backpropagation in neural networks**

---

So,

\[
dL = \frac{dL}{dz^{l-1}} \quad dL = w^{(l)}
\]
