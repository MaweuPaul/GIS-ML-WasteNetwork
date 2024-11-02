# emit_progress(session_id, "Starting waste collection optimization", socketio)

#             result = optimize_waste_collection(engine, session_id, socketio, nyeri_gdf)

#             if result is None:
#                 emit_progress(session_id, "Waste collection optimization failed or returned no results.", socketio)
#             else:
#                 optimal_locations, plot_paths = result
                
#                 if optimal_locations is not None and not optimal_locations.empty:
#                     emit_progress(session_id, f"Waste collection optimization completed. Found {len(optimal_locations)} optimal locations.", socketio)
#                     # Add the optimal locations plot to the buffer_images dictionary
#                     buffer_images['OptimalWasteCollectionPoints'] = plot_paths.get('optimal_locations')
#                     # Add other waste collection plots to the buffer_images dictionary
#                     buffer_images['WasteCollectionRoadSuitability'] = plot_paths.get('road_suitability')
#                     buffer_images['WasteCollectionSettlementSuitability'] = plot_paths.get('settlement_suitability')
#                     buffer_images['WasteCollectionCombinedSuitability'] = plot_paths.get('combined_suitability')
#                 else:
#                     emit_progress(session_id, "Waste collection optimization completed but found no optimal locations.", socketio)