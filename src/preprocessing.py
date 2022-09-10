#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math
import operator
import warnings

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pygeos
import shapely
from packaging.version import Version
from tqdm.auto import tqdm

import networkx as nx
from copy import deepcopy
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, split

from .shape import CircularCompactness

def consolidate_intersections(
    graph,
    tolerance=30,
    rebuild_graph=True,
    rebuild_edges_method="spider",
    directed=None,
    x_col="x",
    y_col="y",
    edge_from_col="from",
    edge_to_col="to",
):
    """
    Consolidate close street intersections into a single node, collapsing short edges.

    If rebuild_graph is True, new edges are drawn according to rebuild_edges_method which is one of:

    1. Extension reconstruction:
        Edges are linearly extended from original endpoints until the new nodes. This method preserves
        most faithfully the network geometry.
    2. Spider-web reconstruction:
        Edges are cropped within a buffer of the new endpoints and linearly extended from there. This
        method improves upon linear reconstruction by mantaining, when possible, network planarity.
    3. Euclidean reconstruction:
        Edges are ignored and new edges are built as straightlines between new origin and new
        destination. This method ignores geometry, but efficiently preserves adjacency.

    If rebuild_graph is False, graph is returned with consolidated nodes but without reconstructed
    edges i.e. graph is intentionally disconnected.

    Graph must be configured so that

    1. All nodes have attributes determining their x and y coordinates;
    2. All edges have attributes determining their origin, destination, and geometry.

    Parameters
    ----------
    graph : Networkx.MultiGraph or Networkx.MultiDiGraph
    tolerance : float
        distance in network units below which nodes will be consolidated
    rebuild_graph: Boolean
    rebuild_edges_method: string
        'extension' or 'spider' or 'euclidean'
    directed: Boolean or None
        consider the graph a MultiDiGraph if True or MultiGraph if False, and
        if None infer from the passed object type
    x_col, y_col: string
        node attribute with the valid coordinate
    edge_from_col, edge_to_col: string
        edge attribute with the valid origin/destination node id

    Returns
    ----------
    Networkx.MultiGraph or Networkx.MultiDiGraph

    """

    # Collect nodes and their data:
    nodes, nodes_dict = zip(*graph.nodes(data=True))
    nodes_df = pd.DataFrame(nodes_dict, index=nodes)
    nodes_geometries = gpd.points_from_xy(nodes_df[x_col], nodes_df[y_col])
    graph_crs = graph.graph.get("crs")
    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        crs=graph_crs,
        geometry=nodes_geometries,
    )

    # In case we did not specify directionality, we infer it from the network:
    if directed is None:
        directed = True if isinstance(graph, nx.MultiDiGraph) else False

    # Create a graph without the edges above a certain length and clean it
    #  from isolated nodes (the unsimplifiable nodes):
    components_graph = deepcopy(graph)
    if not directed:
        components_graph = nx.MultiGraph(components_graph)
    components_graph.remove_edges_from(
        [
            edge
            for edge in graph.edges(keys=True, data=True)
            if edge[-1]["length"] > tolerance
        ]
    )
    isolated_nodes_list = list(nx.isolates(components_graph))
    components_graph.remove_nodes_from(isolated_nodes_list)

    # The connected components of this graph are node clusters we must individually
    #  simplify. We collect them in a dataframe and retrieve node properties (x, y
    #  coords mainly) from the original graph.
    components = nx.connected_components(components_graph)
    components_dict = dict(enumerate(components, start=max(nodes) + 1))
    nodes_to_merge_dict = {
        node: cpt for cpt, nodes in components_dict.items() for node in nodes
    }
    new_nodes_df = pd.DataFrame.from_dict(
        nodes_to_merge_dict, orient="index", columns=["cluster"]
    )
    nodes_to_merge_df = pd.concat(
        [new_nodes_df, nodes_df[[x_col, y_col]]], axis=1, join="inner"
    )

    # The two node attributes we need for the clusters are the position of the cluster
    #  centroids. Those are obtained by averaging the x and y columns. We also add
    # . attribtues referring to the original node ids in every cluster:
    cluster_centroids_df = nodes_to_merge_df.groupby("cluster").mean()
    cluster_centroids_df["simplified"] = True
    cluster_centroids_df["original_node_ids"] = cluster_centroids_df.index.map(
        components_dict
    )
    cluster_geometries = gpd.points_from_xy(
        cluster_centroids_df[x_col], cluster_centroids_df[y_col]
    )
    cluster_gdf = gpd.GeoDataFrame(
        cluster_centroids_df, crs=graph_crs, geometry=cluster_geometries
    )
    cluster_nodes_list = list(cluster_gdf.to_dict("index").items())
    
    # Rebuild edges if necessary:
    if rebuild_graph:
        graph.graph["approach"] = "primal"
        edges_gdf = mm.nx_to_gdf(graph, points=False, lines=True)
        simplified_edges = _get_rebuilt_edges(
            edges_gdf,
            nodes_to_merge_dict,
            cluster_gdf,
            method=rebuild_edges_method,
            buffer=1.5 * tolerance,
            edge_from_col=edge_from_col,
            edge_to_col=edge_to_col,
        )

    # Replacing the collapsed nodes with centroids and adding edges:
    simplified_graph = graph.copy()
    if not directed:
        simplified_graph = nx.MultiGraph(simplified_graph)
    simplified_graph.remove_nodes_from(nodes_to_merge_df.index)
    simplified_graph.add_nodes_from(cluster_nodes_list)

    if rebuild_graph:
        simplified_graph.add_edges_from(simplified_edges)

    return simplified_graph


def _get_rebuilt_edges(
    edges_gdf,
    nodes_dict,
    cluster_gdf,
    method="spider",
    buffer=45,
    edge_from_col="from",
    edge_to_col="to",
):
    """
    Update origin and destination on network edges when original endpoints were replaced by a
      consolidated node cluster. New edges are drawn according to method which is one of:

    1. Extension reconstruction:
        Edges are linearly extended from original endpoints until the new nodes. This method preserves
        most faithfully the network geometry.
    2. Spider-web reconstruction:
        Edges are cropped within a buffer of the new endpoints and linearly extended from there. This
        method improves upon linear reconstruction by mantaining, when possible, network planarity.
    3. Euclidean reconstruction:
        Edges are ignored and new edges are built as straightlines between new origin and new
        destination. This method ignores geometry, but efficiently preserves adjacency.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        GeoDataFrame containing LineString geometry and columns determining origin
        and destination node ids
    nodes_dict: dict
        Dictionary whose keys are node ids and values are the corresponding consolidated
        node cluster ids. Only consolidated nodes are in the dictionary.
    cluster_gdf : GeoDataFrame
        GeoDataFrame containing consolidated node ids.
    method: string
        'extension' or 'spider' or 'euclidean'
    buffer : float
        distance to buffer consolidated nodes in the Spider-web reconstruction
    edge_from_col, edge_to_col: string
        edge attribute with the valid origin/destination node id

    Returns
    ----------
    List
        list of edges that should be added to the network. Edges are in the format
        (origin_id, destination_id, data), where data is inferred from edges_gdf

    """
    # Determine what endpoints were made into clusters:
    edges_gdf["origin_cluster"] = edges_gdf[edge_from_col].apply(
        lambda u: nodes_dict[u] if u in nodes_dict else -1
    )
    edges_gdf["destination_cluster"] = edges_gdf[edge_to_col].apply(
        lambda v: nodes_dict[v] if v in nodes_dict else -1
    )

    # Determine what edges need to be simplified (either between diff.
    #  clusters or self-loops in a cluster):
    edges_tosimplify_gdf = edges_gdf.query(
        f"origin_cluster != destination_cluster or (('{edge_to_col}' == '{edge_from_col}') and origin_cluster >= 0)"
    )

    # Determine the new point geometries (when exists):
    edges_tosimplify_gdf = edges_tosimplify_gdf.assign(
        new_origin_pt=edges_tosimplify_gdf.origin_cluster.map(
            cluster_gdf.geometry, None
        )
    )
    edges_tosimplify_gdf = edges_tosimplify_gdf.assign(
        new_destination_pt=edges_tosimplify_gdf.destination_cluster.map(
            cluster_gdf.geometry, None
        )
    )

    # Determine the new geometry according to the simplification method:
    if method == "extend":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _extension_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    elif method == "euclidean":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _euclidean_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    elif method == "spider":
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _spider_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt, buffer
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )
    else:
        print("Simplification method not recognized. Using spider-web simplification.")
        edges_simplified_geometries = edges_tosimplify_gdf.apply(
            lambda edge: _spider_simplification(
                edge.geometry, edge.new_origin_pt, edge.new_destination_pt, buffer
            ),
            axis=1,
        )
        edges_simplified_gdf = edges_tosimplify_gdf.assign(
            new_geometry=edges_simplified_geometries
        )

    # Rename and update the columns:
    cols_rename = {
        edge_from_col: "original_from",
        edge_to_col: "original_to",
        "origin_cluster": edge_from_col,
        "destination_cluster": edge_to_col,
        "geometry": "original_geometry",
        "new_geometry": "geometry",
    }
    cols_drop = ["new_origin_pt", "new_destination_pt"]
    new_edges_gdf = edges_simplified_gdf.rename(cols_rename, axis=1).drop(
        columns=cols_drop
    )
    new_edges_gdf.loc[:, "length"] = new_edges_gdf.length

    # Update the indices:
    new_edges_gdf.loc[:, edge_from_col] = new_edges_gdf[edge_from_col].where(
        new_edges_gdf[edge_from_col] >= 0, new_edges_gdf["original_from"]
    )
    new_edges_gdf.loc[:, edge_to_col] = new_edges_gdf[edge_to_col].where(
        new_edges_gdf[edge_to_col] >= 0, new_edges_gdf["original_to"]
    )

    # Get the edge list with (from, to, data):
    new_edges_list = list(
        zip(
            new_edges_gdf[edge_from_col],
            new_edges_gdf[edge_to_col],
            new_edges_gdf.iloc[:, 2:].to_dict("index").values(),
        )
    )

    return new_edges_list


def _extension_simplification(geometry, new_origin, new_destination):
    """
    Extends edge geometry to new endpoints.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin, new_destination: shapely.Point or None

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop the line has no endpoints:
    if new_origin == new_destination:
        current_node = Point(line_coords[0])
        geometry = linemerge([LineString([new_origin, current_node]), geometry])
    # Assuming the line is not closed, we can find its endpoints:
    else:
        current_origin, current_destination = geometry.boundary
        if new_origin is not None:
            geometry = linemerge([LineString([new_origin, current_origin]), geometry])
        if new_destination is not None:
            geometry = linemerge(
                [geometry, LineString([current_destination, new_destination])]
            )
    return geometry


def _spider_simplification(geometry, new_origin, new_destination, buff=15):
    """
    Extends edge geometry to new endpoints via a "spider-web" method. Breaks
      current geometry within a buffer of the new endpoint and then extends
      it linearly. Useful to maintain planarity.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin, new_destination: shapely.Point or None
    buff : float
        distance from new endpoint to break current geometry

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop the line has no boundary
    # . and we just use the first coordinate:
    if new_origin == new_destination:
        current_node = Point(line_coords[0])
        geometry = linemerge([LineString([new_origin, current_node]), geometry])
    # Assuming the line is not closed, we can find its endpoints
    #  via the boundary attribute:
    else:
        current_origin, current_destination = geometry.boundary
        if new_origin is not None:
            # Create a buffer around the new origin:
            new_origin_buffer = new_origin.buffer(buff)
            # Use shapely.ops.split to break the edge where it
            #  intersects the buffer:
            geometry_split_by_buffer_list = list(split(geometry, new_origin_buffer))
            # If only one geometry results, edge does not intersect
            #  buffer and line should connect new origin to old origin
            if len(geometry_split_by_buffer_list) == 1:
                geometry_split_by_buffer = geometry_split_by_buffer_list[0]
                splitting_point = current_origin
            # If more than one geometry, merge all linestrings
            #  but the first and get their origin
            else:
                geometry_split_by_buffer = linemerge(geometry_split_by_buffer_list[1:])
                splitting_point = geometry_split_by_buffer.boundary[0]
            # Merge this into new geometry:
            additional_line = [LineString([new_origin, splitting_point])]
            #Consider MultiLineStrings separately:
            if geometry_split_by_buffer.geom_type == 'MultiLineString':
                geometry = linemerge(additional_line + [line for line in geometry_split_by_buffer.geoms])
            else:
                geometry = linemerge(additional_line + [geometry_split_by_buffer])
                
        if new_destination is not None:
            # Create a buffer around the new destination:
            new_destination_buffer = new_destination.buffer(buff)
            # Use shapely.ops.split to break the edge where it
            #  intersects the buffer:
            geometry_split_by_buffer_list = list(
                split(geometry, new_destination_buffer)
            )
            # If only one geometry results, edge does not intersect
            # . buffer and line should connect new destination to old destination
            if len(geometry_split_by_buffer_list) == 1:
                geometry_split_by_buffer = geometry_split_by_buffer_list[0]
                splitting_point = current_destination
            # If more than one geometry, merge all linestrings
            #  but the last and get their destination
            else:
                geometry_split_by_buffer = linemerge(geometry_split_by_buffer_list[:-1])
                splitting_point = geometry_split_by_buffer.boundary[1]
            # Merge this into new geometry:
            additional_line = [LineString([splitting_point, new_destination])]
            #Consider MultiLineStrings separately:
            if geometry_split_by_buffer.geom_type == 'MultiLineString':
                geometry = linemerge([line for line in geometry_split_by_buffer.geoms] + additional_line)
            else:
                geometry = linemerge([geometry_split_by_buffer] + additional_line)
                
    return geometry


def _euclidean_simplification(geometry, new_origin, new_destination):
    """
    Rebuilds edge geometry to new endpoints. Ignores current geometry
      and traces a straight line between new endpoints.

    If either new_origin or new_destination is None, maintains the
      respective current endpoint.

    Parameters
    ----------
    geometry : shapely.LineString
    new_origin, new_destination: shapely.Point or None

    Returns
    ----------
    shapely.LineString

    """
    # If we are dealing with a self-loop, geometry will be null!
    if new_origin == new_destination:
        geometry = None
    # Assuming the line is not closed, we can find its endpoints:
    else:
        current_origin, current_destination = geometry.boundary
        if new_origin is not None:
            if new_destination is not None:
                geometry = LineString([new_origin, new_destination])
            else:
                geometry = LineString([new_origin, current_destination])
        else:
            if new_destination is not None:
                geometry = LineString([current_origin, new_destination])
    return geometry