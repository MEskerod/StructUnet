from itertools import repeat

import numpy as np
##########################################################################################
    


def matching_dict_to_set(matching):
    """Converts matching dict format to matching set format

    Converts a dictionary representing a matching (as returned by
    :func:`max_weight_matching`) to a set representing a matching (as
    returned by :func:`maximal_matching`).

    In the definition of maximal matching adopted by NetworkX,
    self-loops are not allowed, so the provided dictionary is expected
    to never have any mapping from a key to itself. However, the
    dictionary is expected to have mirrored key/value pairs, for
    example, key ``u`` with value ``v`` and key ``v`` with value ``u``.

    """
    edges = set()
    for edge in matching.items():
        u, v = edge
        if (v, u) in edges or edge in edges:
            continue
        edges.add(edge)
    return edges


def max_weight_matching_matrix(G: np.array):
    """Compute a maximum-weighted matching of G.

    A matching is a subset of edges in which no node occurs more than once.
    The weight of a matching is the sum of the weights of its edges.
    A maximal matching cannot add more edges and still be a matching.
    The cardinality of a matching is the number of matched edges.

    Parameters
    ----------
    G : Graph

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    
    Notes
    -----
    This function takes time O(number_of_nodes ** 3).

    If all edge weights are integers, the algorithm uses only integer
    computations.  If floating point weights are used, the algorithm
    could return a slightly suboptimal matching due to numeric
    precision errors.

    This method is based on the "blossom" method for finding augmenting
    paths and the "primal-dual" method for finding a matching of maximum
    weight, both methods invented by Jack Edmonds [1]_.
    A C program for maximum weight matching by Ed Rothberg was used 
    extensively to validate this new code.

    References
    ----------
    .. [1] "Efficient Algorithms for Finding Maximum Matching in Graphs",
       Zvi Galil, ACM Computing Surveys, 1986.
    """
    class NoNode:
        """Dummy value which is different from any node."""

    class Blossom:
        """Representation of a non-trivial blossom or sub-blossom."""

        __slots__ = ["childs", "edges", "mybestedges"]

        # Generate the blossom's leaf vertices.
        def leaves(self):
            stack = [*self.childs]
            while stack:
                t = stack.pop()
                if isinstance(t, Blossom):
                    stack.extend(t.childs)
                else:
                    yield t

    n = G.shape[0]
    neighbors = {i:{j:G[i, j] for j in range(n) if G[i,j] != 0} for i in range(n)}
    # Get a list of vertices.
    gnodes = list(range(n))
    
    if not gnodes:
        return set()  # don't bother with empty graphs

    # Find the maximum edge weight.
    maxweight = np.max(G)

    mate = {}
    label = {}
    labeledge = {}
    inblossom = dict(zip(gnodes, gnodes))
    blossomparent = dict(zip(gnodes, repeat(None)))
    blossombase = dict(zip(gnodes, gnodes))
    bestedge = {}
    dualvar = dict(zip(gnodes, repeat(maxweight)))
    blossomdual = {}
    allowedge = {}
    queue = []

    def slack(v, w):
        """Return 2 * slack of edge (v, w) (does not work inside blossoms)."""
        return dualvar[v] + dualvar[w] - 2 * G[v, w]

    def assignLabel(w, t, v):
        """Assign label t to the top-level blossom containing vertex w, coming through an edge from vertex v."""
        b = inblossom[w]
        assert label.get(w) is None and label.get(b) is None
        label[w] = label[b] = t
        if v is not None:
            labeledge[w] = labeledge[b] = (v, w)
        else:
            labeledge[w] = labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            # b became an S-vertex/blossom; add it(s vertices) to the queue.
            if isinstance(b, Blossom):
                queue.extend(b.leaves())
            else:
                queue.append(b)
        elif t == 2:
            # b became a T-vertex/blossom; assign label S to its mate. (If b is a non-trivial blossom, its base is the only vertex with an external mate.)
            base = blossombase[b]
            assignLabel(mate[base], 1, base)

    def scanBlossom(v, w):
        """Trace back from vertices v and w to discover either a new blossom or an augmenting path. Return the base vertex of the new blossom, or NoNode if an augmenting path was found."""
        # Trace back from v and w, placing breadcrumbs as we go.
        path = []
        base = NoNode
        while v is not NoNode:
            # Look for a breadcrumb in v's blossom or put a new breadcrumb.
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            # Trace one step back.
            if labeledge[b] is None:
                # The base of blossom b is single; stop tracing this path.
                assert blossombase[b] not in mate
                v = NoNode
            else:
                assert labeledge[b][0] == mate[blossombase[b]]
                v = labeledge[b][0]
                b = inblossom[v]
                assert label[b] == 2
                # b is a T-blossom; trace one more step back.
                v = labeledge[b][0]
            # Swap v and w so that we alternate between both paths.
            if w is not NoNode:
                v, w = w, v
        # Remove breadcrumbs.
        for b in path:
            label[b] = 1
        # Return base vertex, if we found one.
        return base

    def addBlossom(base, v, w):
        """Construct a new blossom with given base, through S-vertices v and w.
        Label the new blossom as S; set its dual variable to zero; relabel its T-vertices to S and add them to the queue."""
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        # Create blossom.
        b = Blossom()
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        # Make list of sub-blossoms and their interconnecting edge endpoints.
        b.childs = path = []
        b.edges = edgs = [(v, w)]
        # Trace back from v to base.
        while bv != bb:
            # Add bv to the new blossom.
            blossomparent[bv] = b
            path.append(bv)
            edgs.append(labeledge[bv])
            assert label[bv] == 2 or (
                label[bv] == 1 and labeledge[bv][0] == mate[blossombase[bv]]
            )
            # Trace one step back.
            v = labeledge[bv][0]
            bv = inblossom[v]
        # Add base sub-blossom; reverse lists.
        path.append(bb)
        path.reverse()
        edgs.reverse()
        # Trace back from w to base.
        while bw != bb:
            # Add bw to the new blossom.
            blossomparent[bw] = b
            path.append(bw)
            edgs.append((labeledge[bw][1], labeledge[bw][0]))
            assert label[bw] == 2 or (
                label[bw] == 1 and labeledge[bw][0] == mate[blossombase[bw]]
            )
            # Trace one step back.
            w = labeledge[bw][0]
            bw = inblossom[w]
        # Set label to S.
        assert label[bb] == 1
        label[b] = 1
        labeledge[b] = labeledge[bb]
        # Set dual variable to zero.
        blossomdual[b] = 0
        # Relabel vertices.
        for v in b.leaves():
            if label[inblossom[v]] == 2:
                # This T-vertex now turns into an S-vertex because it becomes part of an S-blossom; add it to the queue.
                queue.append(v)
            inblossom[v] = b
        # Compute b.mybestedges.
        bestedgeto = {}
        for bv in path:
            if isinstance(bv, Blossom):
                if bv.mybestedges is not None:
                    # Walk this subblossom's least-slack edges.
                    nblist = bv.mybestedges
                    # The sub-blossom won't need this data again.
                    bv.mybestedges = None
                else:
                    # This subblossom does not have a list of least-slack edges; get the information from the vertices.
                    nblist = [
                        (v, w) for v in bv.leaves() for w in neighbors[v] if v != w
                    ]
            else:
                nblist = [(bv, w) for w in neighbors[bv] if bv != w]
            for k in nblist:
                (i, j) = k
                if inblossom[j] == b:
                    i, j = j, i
                bj = inblossom[j]
                if (
                    bj != b
                    and label.get(bj) == 1
                    and ((bj not in bestedgeto) or slack(i, j) < slack(*bestedgeto[bj]))
                ):
                    bestedgeto[bj] = k
            # Forget about least-slack edge of the subblossom.
            bestedge[bv] = None
        b.mybestedges = list(bestedgeto.values())
        # Select bestedge[b].
        mybestedge = None
        bestedge[b] = None
        for k in b.mybestedges:
            kslack = slack(*k)
            if mybestedge is None or kslack < mybestslack:
                mybestedge = k
                mybestslack = kslack
        bestedge[b] = mybestedge

    def expandBlossom(b, endstage):
        """Expand the given top-level blossom."""
        # This is an obnoxiously complicated recursive function for the sake of a stack-transformation.  So, we hack around the complexity by using a trampoline pattern. 
        # By yielding the arguments to each recursive call, we keep the actual callstack flat.

        def _recurse(b, endstage):
            # Convert sub-blossoms into top-level blossoms.
            for s in b.childs:
                blossomparent[s] = None
                if isinstance(s, Blossom):
                    if endstage and blossomdual[s] == 0:
                        # Recursively expand this sub-blossom.
                        yield s
                    else:
                        for v in s.leaves():
                            inblossom[v] = s
                else:
                    inblossom[s] = s
            # If we expand a T-blossom during a stage, its sub-blossoms must be
            # relabeled.
            if (not endstage) and label.get(b) == 2:
                # Start at the sub-blossom through which the expanding blossom obtained its label, and relabel sub-blossoms untill we reach the base.
                # Figure out through which sub-blossom the expanding blossom obtained its label initially.
                entrychild = inblossom[labeledge[b][1]]
                # Decide in which direction we will go round the blossom.
                j = b.childs.index(entrychild)
                if j & 1:
                    # Start index is odd; go forward and wrap.
                    j -= len(b.childs)
                    jstep = 1
                else:
                    # Start index is even; go backward.
                    jstep = -1
                # Move along the blossom until we get to the base.
                v, w = labeledge[b]
                while j != 0:
                    # Relabel the T-sub-blossom.
                    if jstep == 1:
                        p, q = b.edges[j]
                    else:
                        q, p = b.edges[j - 1]
                    label[w] = None
                    label[q] = None
                    assignLabel(w, 2, v)
                    # Step to the next S-sub-blossom and note its forward edge.
                    allowedge[(p, q)] = allowedge[(q, p)] = True
                    j += jstep
                    if jstep == 1:
                        v, w = b.edges[j]
                    else:
                        w, v = b.edges[j - 1]
                    # Step to the next T-sub-blossom.
                    allowedge[(v, w)] = allowedge[(w, v)] = True
                    j += jstep
                # Relabel the base T-sub-blossom WITHOUT stepping through to its mate (so don't call assignLabel).
                bw = b.childs[j]
                label[w] = label[bw] = 2
                labeledge[w] = labeledge[bw] = (v, w)
                bestedge[bw] = None
                # Continue along the blossom until we get back to entrychild.
                j += jstep
                while b.childs[j] != entrychild:
                    # Examine the vertices of the sub-blossom to see whether it is reachable from a neighbouring S-vertex outside the expanding blossom.
                    bv = b.childs[j]
                    if label.get(bv) == 1:
                        # This sub-blossom just got label S through one of its neighbours; leave it be.
                        j += jstep
                        continue
                    if isinstance(bv, Blossom):
                        for v in bv.leaves():
                            if label.get(v):
                                break
                    else:
                        v = bv
                    # If the sub-blossom contains a reachable vertex, assign label T to the sub-blossom.
                    if label.get(v):
                        assert label[v] == 2
                        assert inblossom[v] == bv
                        label[v] = None
                        label[mate[blossombase[bv]]] = None
                        assignLabel(v, 2, labeledge[v][0])
                    j += jstep
            # Remove the expanded blossom entirely.
            label.pop(b, None)
            labeledge.pop(b, None)
            bestedge.pop(b, None)
            del blossomparent[b]
            del blossombase[b]
            del blossomdual[b]

        # Now, we apply the trampoline pattern.  We simulate a recursive callstack by maintaining a stack of generators, each yielding a sequence of function arguments.  We grow the stack by appending a call
        # to _recurse on each argument tuple, and shrink the stack whenever a generator is exhausted.
        stack = [_recurse(b, endstage)]
        while stack:
            top = stack[-1]
            for s in top:
                stack.append(_recurse(s, endstage))
                break
            else:
                stack.pop()

    def augmentBlossom(b, v):
        """Swap matched/unmatched edges over an alternating path through blossom b
          between vertex v and the base vertex. Keep blossom bookkeeping consistent."""
        # This is an obnoxiously complicated recursive function for the sake of a stack-transformation.  So, we hack around the complexity by using
        # a trampoline pattern.  By yielding the arguments to each recursive call, we keep the actual callstack flat.

        def _recurse(b, v):
            # Bubble up through the blossom tree from vertex v to an immediate sub-blossom of b.
            t = v
            while blossomparent[t] != b:
                t = blossomparent[t]
            # Recursively deal with the first sub-blossom.
            if isinstance(t, Blossom):
                yield (t, v)
            # Decide in which direction we will go round the blossom.
            i = j = b.childs.index(t)
            if i & 1:
                # Start index is odd; go forward and wrap.
                j -= len(b.childs)
                jstep = 1
            else:
                # Start index is even; go backward.
                jstep = -1
            # Move along the blossom until we get to the base.
            while j != 0:
                # Step to the next sub-blossom and augment it recursively.
                j += jstep
                t = b.childs[j]
                if jstep == 1:
                    w, x = b.edges[j]
                else:
                    x, w = b.edges[j - 1]
                if isinstance(t, Blossom):
                    yield (t, w)
                # Step to the next sub-blossom and augment it recursively.
                j += jstep
                t = b.childs[j]
                if isinstance(t, Blossom):
                    yield (t, x)
                # Match the edge connecting those sub-blossoms.
                mate[w] = x
                mate[x] = w
            # Rotate the list of sub-blossoms to put the new base at the front.
            b.childs = b.childs[i:] + b.childs[:i]
            b.edges = b.edges[i:] + b.edges[:i]
            blossombase[b] = blossombase[b.childs[0]]
            assert blossombase[b] == v

        # Now, we apply the trampoline pattern.  We simulate a recursive callstack by maintaining a stack of generators, each yielding a sequence of function arguments.  We grow the stack by appending a call
        # to _recurse on each argument tuple, and shrink the stack whenever a generator is exhausted.
        stack = [_recurse(b, v)]
        while stack:
            top = stack[-1]
            for args in top:
                stack.append(_recurse(*args))
                break
            else:
                stack.pop()

    def augmentMatching(v, w):
        """Swap matched/unmatched edges over an alternating path between two single vertices. The augmenting path runs through S-vertices v and w."""
        for s, j in ((v, w), (w, v)):
            # Match vertex s to vertex j. Then trace back from s until we find a single vertex, swapping matched and unmatched edges as we go.
            while 1:
                bs = inblossom[s]
                assert label[bs] == 1
                assert (labeledge[bs] is None and blossombase[bs] not in mate) or (
                    labeledge[bs][0] == mate[blossombase[bs]]
                )
                # Augment through the S-blossom from s to base.
                if isinstance(bs, Blossom):
                    augmentBlossom(bs, s)
                # Update mate[s]
                mate[s] = j
                # Trace one step back.
                if labeledge[bs] is None:
                    # Reached single vertex; stop.
                    break
                t = labeledge[bs][0]
                bt = inblossom[t]
                assert label[bt] == 2
                # Trace one more step back.
                s, j = labeledge[bt]
                # Augment through the T-blossom from j to base.
                assert blossombase[bt] == t
                if isinstance(bt, Blossom):
                    augmentBlossom(bt, j)
                # Update mate[j]
                mate[j] = s

    
    # Main loop: continue until no further improvement is possible.
    while 1:
        # Each iteration of this loop is a "stage".
        # A stage finds an augmenting path and uses that to improve the matching.

        # Remove labels from top-level blossoms/vertices.
        label.clear()
        labeledge.clear()

        # Forget all about least-slack edges.
        bestedge.clear()
        for b in blossomdual:
            b.mybestedges = None

        # Loss of labeling means that we can not be sure that currently allowable edges remain allowable throughout this stage.
        allowedge.clear()

        # Make queue empty.
        queue[:] = []

        # Label single blossoms/vertices with S and put them in the queue.
        for v in gnodes:
            if (v not in mate) and label.get(inblossom[v]) is None:
                assignLabel(v, 1, None)

        # Loop until we succeed in augmenting the matching.
        augmented = 0
        while 1:
            # Each iteration of this loop is a "substage".
            # A substage tries to find an augmenting path; if found, the path is used to improve the matching and the stage ends. 
            # If there is no augmenting path, the primal-dual method is used to pump some slack out of the dual variables.

            # Continue labeling until all vertices which are reachable through an alternating path have got a label.
            while queue and not augmented:
                # Take an S vertex from the queue.
                v = queue.pop()
                assert label[inblossom[v]] == 1

                # Scan its neighbours:
                for w in neighbors[v]:
                    if w == v:
                        continue  # ignore self-loops
                    # w is a neighbour to v
                    bv = inblossom[v]
                    bw = inblossom[w]
                    if bv == bw:
                        # this edge is internal to a blossom; ignore it
                        continue
                    if (v, w) not in allowedge:
                        kslack = slack(v, w)
                        if kslack <= 0:
                            # edge k has zero slack => it is allowable
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    if (v, w) in allowedge:
                        if label.get(bw) is None:
                            # (C1) w is a free vertex; label w with T and label its mate with S (R12).
                            assignLabel(w, 2, v)
                        elif label.get(bw) == 1:
                            # (C2) w is an S-vertex (not in the same blossom); follow back-links to discover either an augmenting path or a new blossom.
                            base = scanBlossom(v, w)
                            if base is not NoNode:
                                # Found a new blossom; add it to the blossom bookkeeping and turn it into an S-blossom.
                                addBlossom(base, v, w)
                            else:
                                # Found an augmenting path; augment the matching and end this stage.
                                augmentMatching(v, w)
                                augmented = 1
                                break
                        elif label.get(w) is None:
                            # w is inside a T-blossom, but w itself has not yet been reached from outside the blossom;
                            # mark it as reached (we need this to relabel during T-blossom expansion).
                            assert label[bw] == 2
                            label[w] = 2
                            labeledge[w] = (v, w)
                    elif label.get(bw) == 1:
                        # keep track of the least-slack non-allowable edge to a different S-blossom.
                        if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
                            bestedge[bv] = (v, w)
                    elif label.get(w) is None:
                        # w is a free vertex (or an unreached vertex inside a T-blossom) but we can not reach it yet; keep track of the least-slack edge that reaches w.
                        if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
                            bestedge[w] = (v, w)

            if augmented:
                break

            # There is no augmenting path under these constraints; compute delta and reduce slack in the optimization problem.
            # (Note that our vertex dual variables, edge slacks and delta's are pre-multiplied by two.)
            deltatype = -1
            delta = deltaedge = deltablossom = None

            # Compute delta1: the minimum value of any vertex dual.
            deltatype = 1
            delta = min(dualvar.values())

            # Compute delta2: the minimum slack on any edge between an S-vertex and a free vertex.
            for v in gnodes:
                if label.get(inblossom[v]) is None and bestedge.get(v) is not None:
                    d = slack(*bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]

            # Compute delta3: half the minimum slack on any edge between a pair of S-blossoms.
            for b in blossomparent:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 1
                    and bestedge.get(b) is not None
                ):
                    kslack = slack(*bestedge[b])
                    d = kslack / 2.0
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]

            # Compute delta4: minimum z variable of any T-blossom.
            for b in blossomdual:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 2
                    and (deltatype == -1 or blossomdual[b] < delta)
                ):
                    delta = blossomdual[b]
                    deltatype = 4
                    deltablossom = b

            if deltatype == -1:
                # No further improvement possible; max-cardinality optimum reached. Do a final delta update to make the optimum verifiable.
                deltatype = 1
                delta = max(0, min(dualvar.values()))

            # Update dual variables according to delta.
            for v in gnodes:
                if label.get(inblossom[v]) == 1:
                    # S-vertex: 2*u = 2*u - 2*delta
                    dualvar[v] -= delta
                elif label.get(inblossom[v]) == 2:
                    # T-vertex: 2*u = 2*u + 2*delta
                    dualvar[v] += delta
            for b in blossomdual:
                if blossomparent[b] is None:
                    if label.get(b) == 1:
                        # top-level S-blossom: z = z + 2*delta
                        blossomdual[b] += delta
                    elif label.get(b) == 2:
                        # top-level T-blossom: z = z - 2*delta
                        blossomdual[b] -= delta

            # Take action at the point where minimum delta occurred.
            if deltatype == 1:
                # No further improvement possible; optimum reached.
                break
            elif deltatype == 2:
                # Use the least-slack edge to continue the search.
                (v, w) = deltaedge
                assert label[inblossom[v]] == 1
                allowedge[(v, w)] = allowedge[(w, v)] = True
                queue.append(v)
            elif deltatype == 3:
                # Use the least-slack edge to continue the search.
                (v, w) = deltaedge
                allowedge[(v, w)] = allowedge[(w, v)] = True
                assert label[inblossom[v]] == 1
                queue.append(v)
            elif deltatype == 4:
                # Expand the least-z blossom.
                expandBlossom(deltablossom, False)

            # End of a this substage.

        # Paranoia check that the matching is symmetric.
        for v in mate:
            assert mate[mate[v]] == v

        # Stop when no more augmenting path can be found.
        if not augmented:
            break

        # End of a stage; expand all S-blossoms which have zero dual.
        for b in list(blossomdual.keys()):
            if b not in blossomdual:
                continue  # already expanded
            if blossomparent[b] is None and label.get(b) == 1 and blossomdual[b] == 0:
                expandBlossom(b, True)

    return matching_dict_to_set(mate)